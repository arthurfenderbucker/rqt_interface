import os
import rospy
import rospkg

import json
import numpy as np

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget

from std_msgs.msg import Empty, String, Bool
from orb_slam3_ros.srv import SaveMap
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose, Point, Twist, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path

from scipy.spatial.transform import Rotation

import ros_numpy
import tf

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from copy import deepcopy

from bebop_msgs.msg import CommonCommonStateBatteryStateChanged, Ardrone3PilotingStateFlyingStateChanged

class MyPlugin(Plugin):

    def __init__(self, context):
        super(MyPlugin, self).__init__(context)
        # Give QObjects reasonable names
        self.setObjectName('MyPlugin')

        # Process standalone plugin command-line arguments
        from argparse import ArgumentParser
        parser = ArgumentParser()
        # Add argument(s) to the parser.
        parser.add_argument("-q", "--quiet", action="store_true",
                      dest="quiet",
                      help="Put plugin in silent mode")
        args, unknowns = parser.parse_known_args(context.argv())
        if not args.quiet:
            print('arguments: ', args)
            print('unknowns: ', unknowns)

        # Create QWidget
        self._widget = QWidget()
        # Get path to UI file which should be in the "resource" folder of this package
        ui_file = os.path.join(rospkg.RosPack().get_path('rqt_interface'), 'resource', 'MyPlugin.ui')
        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._widget)
        # Give QObjects reasonable names
        self._widget.setObjectName('MyPluginUi')
        # Show _widget.windowTitle on left-top of each plugin (when 
        # it's set in _widget). This is useful when you open multiple 
        # plugins at once. Also if you open multiple instances of your 
        # plugin at once, these lines add number to make it easy to 
        # tell from pane to pane.
        if context.serial_number() > 1:
            self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))
        # Add widget to the user interface
        context.add_widget(self._widget)



        # ros parameters
        self.map_name = rospy.get_param('~map_name',"demo")
        topic_slam_pose = rospy.get_param('~topic_slam_pose',"/orb_slam3/camera_pose") #/orb_slam2_mono/pose


        self.slam_pose = None 
        self.odom = None
        self.slam_rotation = None
        self.slam_scale_factor = None
        self.running_control = False
        self.goal_pose = None
        self.count_aligned = 0
        self.waypoint_selected = None
        self.waypoint_selected_index = None
        self.flying_state = 0
        self.drone_path = Path()
        self.manual_mode = False
        self.count_map_odom = 0

        #bt callbacks setup
        self._widget.takeoff.clicked.connect(self.takeoff_bt_cb)
        self._widget.land.clicked.connect(self.land_bt_cb)
        self._widget.load.clicked.connect(self.load_bt_cb)
        self._widget.execute.clicked.connect(self.execute_bt_cb)
        self._widget.save_map.clicked.connect(self.save_map_bt_cb)
        self._widget.calibrate_origin.clicked.connect(self.calibrate_origin_bt_cb)
        self._widget.calibrate_scale.clicked.connect(self.calibrate_scale_bt_cb)
        self._widget.calibrate_angle.clicked.connect(self.calibrate_angle_bt_cb)

        self._widget.new_traj.clicked.connect(self.new_traj_bt_cb)
        self._widget.del_traj.clicked.connect(self.del_traj_bt_cb)
        self._widget.del_wp.clicked.connect(self.del_wp_bt_cb)
        self._widget.import_traj.clicked.connect(self.import_traj_bt_cb)
        self._widget.add_wp.clicked.connect(self.add_wp_bt_cb)
        self._widget.edit_wp.clicked.connect(self.edit_wp_bt_cb)



        self._widget.manual_control.clicked.connect(self.manual_control_cb)

        self._widget.up.clicked.connect(lambda: self.manual_commands_cb("up"))
        self._widget.down.clicked.connect(lambda: self.manual_commands_cb("down"))
        self._widget.front.clicked.connect(lambda: self.manual_commands_cb("front"))
        self._widget.back.clicked.connect(lambda: self.manual_commands_cb("back"))
        self._widget.right.clicked.connect(lambda: self.manual_commands_cb("right"))
        self._widget.left.clicked.connect(lambda: self.manual_commands_cb("left"))
        self._widget.rotate_cw.clicked.connect(lambda: self.manual_commands_cb("rotate_cw"))
        self._widget.rotate_acw.clicked.connect(lambda: self.manual_commands_cb("rotate_acw"))



        


        self._widget.custom_action.clicked.connect(self.custom_action_bt_cb)
        self._widget.custom_action.setEnabled(False)




        self._widget.reset_calibration.clicked.connect(self.reset_calibration_bt_cb)
        self._widget.trajectory_list.clicked.connect(self.trajectory_bt_cb)
        self._widget.waypoints_list.clicked.connect(self.waypoints_bt_cb)


        self._widget.map_name.setText("Map name: "+self.map_name)


        self.checklist = {"bebop odom":None,
                         "SLAM":None,
                         "trajectory":None,
                         "map odom":None,
                         "calibrated": None}
        self._widget.list.addItems(self.checklist.keys())

        self.trajectory_selected = None


        #ros publishers
        self._pub_takeoff = rospy.Publisher('takeoff', Empty, queue_size=10)
        self._pub_land = rospy.Publisher('land', Empty, queue_size=10)
        self._pub_execute = rospy.Publisher('execute', Empty, queue_size=10)
        self._pub_custom_action = rospy.Publisher("custom_action", String, queue_size=1)

        self._traj_name_execute = rospy.Publisher('traj_name', String, queue_size=10)
        self._pub_camera_ang = rospy.Publisher("/bebop/camera_control", Twist, queue_size=1)

        #visualization
        self._pub_waypoints = rospy.Publisher("/flight_path_marker", MarkerArray, queue_size=10)
        self._pub_path_pann = rospy.Publisher('/flight_path', Path, latch=True, queue_size=10)
        self._pub_drone_marker = rospy.Publisher("/drone_marker", Marker, queue_size=10)
        self._pub_drone_path = rospy.Publisher("/drone_path", Path, queue_size=10)
        self._pub_goal_pose_marker = rospy.Publisher("/goal_pose_marker", Marker, queue_size=10)

        self._pub_set_manual_mode = rospy.Publisher("/control/set_manual_mode", Bool, queue_size=10)
        self._pub_cmd_vel = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=1)



        self._odom_sub = rospy.Subscriber('/bebop/odom', Odometry, self.odom_cb)
        self._battery_sub = rospy.Subscriber('/bebop/states/common/CommonState/BatteryStateChanged', CommonCommonStateBatteryStateChanged, self.battery_cb)
        self._battery_sub = rospy.Subscriber('/bebop/states/ardrone3/PilotingState/FlyingStateChanged', Ardrone3PilotingStateFlyingStateChanged, self.flying_state_cb)


        self._slam_sub = rospy.Subscriber(topic_slam_pose, PoseStamped, self.slam_cb)
        self.current_pose_sub  = rospy.Subscriber("odom_slam_sf/current_odom", Odometry, self.map_odom_cb)

        self._sub_control_pose = rospy.Subscriber("/control/position", Pose, self.pose_goal_cb)
        self._sub_control_running_state = rospy.Subscriber("/control/set_running_state", Bool, self.running_state_cb)
        self._sub_control_running_state = rospy.Subscriber("/control/aligned", Bool, self.alligned_cb)



        self._cached_stamp  = 0
        #load trajecotry list
        rospack = rospkg.RosPack()
        self.trajectories_path = str(rospack.get_path('drone_control')+'/config/recorded_routines.json')
        self.load_moving_routines()


        # odom to slam calibrtation
        rospack = rospkg.RosPack()
        self.calibration_path = str(rospack.get_path('odom_slam_sensor_fusion')+'/config/maps/slam_calibration.json')

        try:
            with open(self.calibration_path, 'r') as json_data_file:
                self.calibration_data = json.load(json_data_file)
        except:
            self.calibration_data = {}
            rospy.logerr("Error loading calibration data")
            self._widget.error.setText("calibration data is empty")


        self.check_calibration()
        #update routine

        self._timer = rospy.Timer(rospy.Duration(0.1), self.update_interface)
    


    # ================================ bt callback ==================================
    def save_map_bt_cb(self):
        print("save_map")

        try:
            save_map_srv = rospy.ServiceProxy('/orb_slam3/save_map', SaveMap)
            save_map_srv(self.map_name)
            print("map saved")
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)


    def takeoff_bt_cb(self):
        print("takeoff")
        self._pub_takeoff.publish(Empty())
    def land_bt_cb(self):
        print("land")
        self._pub_land.publish(Empty())
    def load_bt_cb(self):
        print("load")

        self._widget.trajectory_selected.setText("Trajectory loaded for execution: <b>"+self.trajectory_selected+"</b>")
        self.publish_wp_markers()
        self._traj_name_execute.publish(self.trajectory_selected)

        self.checklist["trajectory"] = self.moving_routines[self.trajectory_selected]

    def execute_bt_cb(self):
        print("execute")
        self.drone_path = Path()
        self._pub_execute.publish(Empty())
        self.publish_wp_markers()

        # self.execute()

    def trajectory_bt_cb(self):
        self.trajectory_selected = self._widget.trajectory_list.currentItem().text()
        self.waypoint_selected = None
        self.waypoint_selected_index = None
        print("trajectory_callback:", self.trajectory_selected)

        self.update_waypoints()

    def waypoints_bt_cb(self):
        print("waypoints_bt_cb")
        #get current waypoint and create a dialog to edit it
        wp_text = self._widget.waypoints_list.currentItem().text()
        index_len = len(wp_text.split(",")[-1])+2
        self.waypoint_selected = wp_text[:-index_len]
        print("waypoint_callback:", self.waypoint_selected)
        self.waypoint_selected_index = self._widget.waypoints_list.currentRow()-1
        print("waypoint_callback:", self.waypoint_selected_index)


      

    def calibrate_origin_bt_cb(self):
        print("calibrate_origin_bt_cb")
        if not self.slam_pose is None:
            if not self.map_name in self.calibration_data.keys():
                self.calibration_data[self.map_name] = {}
            self.calibration_data[self.map_name]["origin_slam_pose"] = ros_numpy.numpify(self.slam_pose.pose).tolist()

            if not self.odom is None:
                # self.calibration_data[self.map_name]["origin_odom_pose"] = ros_numpy.numpify(self.odom.pose.pose).tolist()
                self.odom_rotation = ros_numpy.numpify(self.odom.pose.pose.orientation)
                self.calibration_data[self.map_name]["odom_origin_rotation"] = self.odom_rotation.tolist()

            self.save_calibration()
    
    def calibrate_scale_bt_cb(self):
        print("calibrate_scale_bt_cb")
        if not self.slam_pose is None:
            if not self.map_name in self.calibration_data.keys():
                self.calibration_data[self.map_name] = {}
            self.calibration_data[self.map_name]["scale_slam_pose"] = ros_numpy.numpify(self.slam_pose.pose).tolist()
            self.save_calibration()
    
    def calibrate_angle_bt_cb(self):
        print("calibrate_angle_bt_cb")
        if not self.slam_pose is None:
            if not self.map_name in self.calibration_data.keys():
                self.calibration_data[self.map_name] = {}
            self.calibration_data[self.map_name]["scale_2_slam_pose"] = ros_numpy.numpify(self.slam_pose.pose).tolist()
            self.save_calibration()


    def reset_calibration_bt_cb(self):
        
        if self.map_name in self.calibration_data.keys():
            del self.calibration_data[self.map_name]
            self.checklist["calibrated"] = None
            self.save_calibration()
        else:
            rospy.logerr("No calibration data for this map")

    def new_traj_bt_cb(self):
        print("new_traj_bt_cb")

        text, ok = QInputDialog.getText(self._widget, 'New trajectory', 'Enter the trajectory name:')
        if ok:
            self.moving_routines[text] = []
            self._widget.trajectory_list.addItem(text)
            self.save_moving_routines()
            self.trajectory_selected = text
            self.update_waypoints()

    def edit_wp_bt_cb(self):
        if not self.waypoint_selected_index is None and not self.trajectory_selected is None:
            
            # wp_text = self._widget.waypoints_list.currentItem().text()
            # index_len = len(wp_text.split(",")[-1])+1

            text, ok = QInputDialog.getText(self._widget, 'Waypoint editor', 'Edit the waypoint name:', QLineEdit.Normal, self.waypoint_selected)
            if ok:
                wp = [float(i) for i in text.split(",")]
                if len(wp) == 4:
                    self.moving_routines[self.trajectory_selected][self.waypoint_selected_index]=ros_numpy.numpify(self.wp_to_pose(wp)).tolist()
                    self.update_waypoints()
                    self.save_moving_routines()
                else:
                    rospy.logerr("Error parsing waypoint")
                    self._widget.error.setText("Error parsing waypoint\nuse the format:x,y,z,yaw")
        else:
            rospy.logerr("No waypoint selected")

    def del_wp_bt_cb(self):
        print("del_wp_bt_cb")
        if not self.waypoint_selected_index is None and not self.trajectory_selected is None:
            del self.moving_routines[self.trajectory_selected][self.waypoint_selected_index]
            self.save_moving_routines()
            self.update_waypoints()

        else:
            rospy.logerr("No waypoint selected")


    def del_traj_bt_cb(self):

        print("del_traj_bt_cb")
        selected_traj_index = self._widget.trajectory_list.currentIndex().row()

        item, ok = QInputDialog.getItem(self._widget, "Delete trajectory", 
         "select a routine", self.moving_routines.keys(), selected_traj_index, False)
			
        if ok and item:
            del self.moving_routines[item]
            self.save_moving_routines()
            self._widget.trajectory_list.clear()
            self._widget.trajectory_list.addItems(self.moving_routines.keys())
            self.trajectory_selected = None
            self.update_waypoints()

    def import_traj_bt_cb(self):
        print("import_traj_bt_cb")

        file_name = QFileDialog.getOpenFileName(self._widget, 'Open file',
            self.trajectories_path, "Trajectory files (*.json)") 
        if file_name[0]:
            with open(file_name[0]) as json_file:
                data = json.load(json_file)
                traj_name = file_name[0].split("/")[-1][:-5]
                wps = self.json_to_routine(data)
                self.moving_routines[traj_name] = wps
                self.save_moving_routines()
                self._widget.trajectory_list.clear()
                self._widget.trajectory_list.addItems(self.moving_routines.keys())
                self.trajectory_selected =traj_name
                self.update_waypoints()

    def add_wp_bt_cb(self):
        print("add_wp_bt_cb")
        if not self.slam_pose is None and not self.trajectory_selected is None:
            self.moving_routines[self.trajectory_selected].append(ros_numpy.numpify(self.map_odom.pose.pose).tolist())
            self.save_moving_routines()
            self.update_waypoints()
    
    def custom_action_bt_cb(self):
        print("custom_action_bt_cb")
        self._pub_custom_action.publish("custom")


    def manual_commands_cb(self, dir_key):
        print("manual_commands_cb\t"+dir_key)
        # self._pub_set_manual_mode.publish(True)

        moveBindings = {
            'front':[1,0,0,0],
            'back':[-1,0,0,0],
            'right':[0,-1,0,0],
            'left':[0,1,0,0],
            'up':[0,0,1,0],
            'down':[0,0,-1,0],
            'rotate_cw':[0,0,0,1],
            'rotate_acw':[0,0,0,-1]}
        
        speed = 0.5
        cmd = moveBindings[dir_key]
        vel = Twist()
        vel.linear.x = cmd[0]*speed
        vel.linear.y = cmd[1]*speed
        vel.linear.z = cmd[2]*speed
        vel.angular.z = cmd[3]

        self._pub_cmd_vel.publish(vel)

        vel_zero = Twist()
        vel_zero.linear.x = 0.0
        vel_zero.linear.y = 0.0
        vel_zero.linear.z = 0.0

        rospy.sleep(0.5)
        self._pub_cmd_vel.publish(vel_zero)

        

    def manual_control_cb(self):
        
        self.manual_mode = not self.manual_mode
        self._pub_set_manual_mode.publish(self.manual_mode)

        self._widget.manual_control.setText("disable manual" if self.manual_mode else "enable manual")
    

    # ================================ topic callbacks ==================================

            
    def slam_cb(self,msg):
        self.slam_pose=msg
        self.checklist["SLAM"] = msg.header.stamp

    def odom_cb(self,msg):

        if self.odom is None:
            #align camera with world
            camera_init_angle = Twist()
            camera_init_angle.angular.y = 3 # looking fowards
            self._pub_camera_ang.publish(camera_init_angle)

        self.odom = msg
        self.checklist["bebop odom"] = msg.header.stamp

    def map_odom_cb(self,msg):
        self.map_odom = msg
        self.checklist["map odom"] = msg.header.stamp
        p = ros_numpy.numpify(msg.pose.pose)[:3,3]
        self._widget.drone_pose.setText("pose x:{:.2f}, y:{:.2f}, z{:.2f}".format(p[0],p[1],p[2]))

        self.count_map_odom+=1
        if self.count_map_odom%10 == 0:
            self.publish_drone_marker()

    
    def pose_goal_cb(self, msg):

        self.publish_goal_pose_marker(msg)
        
        
    def running_state_cb(self, msg):
        self.running_control = msg.data
        self.count_aligned = 0
        # if not self.running_control:
        #     self.reset_wp_markers()


    def alligned_cb(self, msg):
        if self.running_control:
            if self.count_aligned<self._widget.waypoints_list.count():
                self._widget.waypoints_list.item(self.count_aligned+1).setBackground( QColor('#7fc97f'))
        self.count_aligned += 1

    def battery_cb(self,msg):
        print("battery_cb")

        self._widget.battery.setText("Battery: "+str(msg.percent))
        
        if msg.percent < 60:
            self._widget.battery.setStyleSheet("QLabel { background-color : yellow; color : black; }")
        if msg.percent < 40:
            self._widget.battery.setStyleSheet("QLabel { background-color : yellow; color : black; }")
        elif msg.percent < 20:
            self._widget.battery.setStyleSheet("QLabel { background-color : red; color : black; }")
    
    def flying_state_cb(self,msg):
        print("flying_cb")
        self.flying_state = msg.state


    # ================================ calibration ==================================

    def check_calibration(self):
        if self.map_name in self.calibration_data.keys():
            self.checklist["calibrated"] = True
            return True
        else:
            self.checklist["calibrated"] = None
            return False


    def compute_tranform(self):
        ref_origin_coord = np.array([self._widget.origin_x.value(),self._widget.origin_y.value(),self._widget.origin_z.value()])
        ref_scale_coord = np.array([self._widget.scale_x.value(),self._widget.scale_y.value(),self._widget.scale_z.value()])
        ref_scale_2_coord = np.array([self._widget.scale_x_2.value(),self._widget.scale_y_2.value(),self._widget.scale_z_2.value()])

        slam_origin_coord = np.array(self.calibration_data[self.map_name]["origin_slam_pose"])
        slam_scale_coord  = np.array(self.calibration_data[self.map_name]["scale_slam_pose"])
        slam_scale_2_coord  = np.array(self.calibration_data[self.map_name]["scale_2_slam_pose"])

        ref_vec = ref_scale_coord[:3]-ref_origin_coord[:3]
        ref_vec_2 = ref_scale_2_coord[:3]-ref_origin_coord[:3]
        slam_vec = slam_scale_coord[:3,3]-slam_origin_coord[:3,3]
        slam_vec_2 = slam_scale_2_coord[:3,3]-slam_origin_coord[:3,3]

        
        #calculate scale factor
        ref_dist = np.linalg.norm(ref_vec)
        slam_dist = np.linalg.norm(slam_vec)
        scale_factor = ref_dist/slam_dist

        self.slam_scale_factor = scale_factor
        self.slam_rotation = Rotation.align_vectors(np.stack((ref_vec,ref_vec_2)),np.stack((slam_vec*scale_factor,slam_vec_2*scale_factor)))[0]
        
        
        self.calibration_data[self.map_name]["rotation"] = self.slam_rotation.as_quat().tolist()
        self.calibration_data[self.map_name]["scale_factor"] = float(scale_factor)
        self.calibration_data[self.map_name]["ref_origin_coord"] = ref_origin_coord.tolist()
        self.calibration_data[self.map_name]["ref_scale_coord"] = ref_scale_coord.tolist()


    def save_calibration(self):
        

        if self.slam_pose is None:
            print("no slam")
            self._widget.error.setText("no slam")
        elif self.map_name in self.calibration_data.keys():
            if not "origin_slam_pose" in self.calibration_data[self.map_name]:
                self._widget.error.setText("origin not calibrated")
            elif not "scale_slam_pose" in self.calibration_data[self.map_name]:
                self._widget.error.setText("scale not calibrated")
            elif not "scale_2_slam_pose" in self.calibration_data[self.map_name]:
                self._widget.error.setText("scale 2 not calibrated")
            elif self.odom is None:
                self._widget.error.setText("no odom data received")
            else:
                self.compute_tranform()
                self.checklist["calibrated"] = True

        with open(self.calibration_path , 'w') as json_data_file:
            json.dump(self.calibration_data, json_data_file)

        print("calibration saved")
    # ================================ settings =================================

    def publish_drone_marker(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()

        marker.mesh_use_embedded_materials = True

        marker.mesh_resource = "package://bebop_description/meshes/bebop_model.stl"

        marker.id = 0
        marker.type = marker.MESH_RESOURCE
        marker.action = marker.ADD

        r,p,y = tf.transformations.euler_from_quaternion(ros_numpy.numpify(self.map_odom.pose.pose.orientation))
        q = ros_numpy.msgify(Quaternion, tf.transformations.quaternion_from_euler(0,0,y+np.pi/2))
        marker.pose.position = self.map_odom.pose.pose.position
        marker.pose.orientation = q
        # marker.pose.orientation = self.map_odom.pose.pose.position

        marker.scale.x = 0.001
        marker.scale.y = 0.001
        marker.scale.z = 0.001
        # marker.color.a = 1.0
        # marker.color.r = 0.0
        # marker.color.g = 1.0
        if self.flying_state != 0:
            self.drone_path.header = self.map_odom.header
            pose = PoseStamped()
            pose.header = self.map_odom.header
            pose.pose = self.map_odom.pose.pose
            self.drone_path.poses.append(pose)
            self._pub_drone_path.publish(self.drone_path)

        self._pub_drone_marker.publish(marker)

    def reset_wp_markers(self):
        
        marker = Marker()
        marker.id = 0
        marker.header.frame_id = "world"
        marker.action = marker.DELETE
        self._pub_goal_pose_marker.publish(marker)


        markerArray = MarkerArray()
        marker2 = Marker()
        marker2.id = 0
        marker2.header.frame_id = "world"
        marker2.action = marker2.DELETE
        markerArray.markers.append(marker2) 
        for i in range(100):

            marker = Marker()
            marker.id = i+1
            marker.header.frame_id = "world"
            marker.action = marker.DELETE
            markerArray.markers.append(marker)
        self._pub_waypoints.publish(markerArray)

    def publish_wp_markers(self):
        markerArray = MarkerArray()
        marker2 = Marker()
        marker2.id = 0
        marker2.lifetime = rospy.Duration()
        marker2.header.frame_id = "world"
        marker2.type = marker2.LINE_STRIP
        marker2.action = marker2.ADD
        marker2.scale.x = 0.05
        marker2.color.a = 1.0
        marker2.color.b = 1.0
        marker2.pose.orientation.w=1.0
        marker2.points=[] 
        markerArray.markers.append(marker2) 
        for i,wp_pose in enumerate(self.moving_routines[self.trajectory_selected]):
            pose =ros_numpy.msgify(Pose,np.array(wp_pose))

            p = Point() 
            p.x = pose.position.x
            p.y = pose.position.y
            p.z = pose.position.z

            marker2.points.append(p)
            
            marker = Marker()
            marker.id = 2*i+1
            marker.header.frame_id = "world"
            marker.type = marker.ARROW
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.b = 1.0
            marker.color.r = 1.0
            marker.color.a = 1.0
            marker.pose = pose

            markerArray.markers.append(marker)
            marker_text = Marker()

            pose_text =ros_numpy.msgify(Pose,np.array(wp_pose))
            marker_text.id = 2*i+2
            marker_text.header.frame_id = "world"
            marker_text.type = marker_text.TEXT_VIEW_FACING
            marker_text.action = marker_text.ADD
            marker_text.text = str(i+1)
            marker_text.pose = pose_text
            marker_text.pose.position.z += 0.1
            marker_text.color.g = 1.0
            marker_text.color.b = 1.0
            marker_text.color.a = 1.0
            marker_text.scale.z = 0.15

            markerArray.markers.append(marker_text)


        self._pub_waypoints.publish(markerArray)

    def publish_goal_pose_marker(self,msg):
        marker = Marker()
        marker.id = 0
        marker.header.frame_id = "world"
        marker.type = marker.ARROW
        marker.action = marker.ADD if self.running_control else marker.DELETE
        marker.scale.x = 0.15
        marker.scale.y = 0.075
        marker.scale.z = 0.075
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose = msg

        self.goal_pose = msg
        self._pub_goal_pose_marker.publish(marker)

    def wp_to_pose(self, wp):
        """Converts a waypoint to a Pose message."""

        p = Pose()
        p.position.x, p.position.y, p.position.z = wp[0], wp[1], wp[2]
        if len(wp) == 4:
            q = tf.transformations.quaternion_from_euler(0,0,wp[3])
            print(q)
            p.orientation = ros_numpy.msgify(Quaternion,q)

        return p

    def json_to_routine(self, json_routine):
        routine = []
        # for _wp in json_routine["mission"]["items"]:
            # wp = _wp["params"]
        for _wp in json_routine["wps"]:
            p = self.wp_to_pose(_wp)
            routine.append(ros_numpy.numpify(p).tolist())

        return routine

    def save_moving_routines(self):
        try:
            with open(self.trajectories_path, 'w') as json_data_file:
                json.dump(self.moving_routines,json_data_file)
        except:
            rospy.logerr("Error saving moving routines")

    def load_moving_routines(self):
        try:
            with open(self.trajectories_path, 'r') as json_data_file:
                self.moving_routines = json.load(json_data_file)
        except:
            self.moving_routines = {}
            rospy.logerr("Error loading moving routines")
        
        self._widget.trajectory_list.clear()
        self._widget.trajectory_list.addItems(self.moving_routines.keys())
        
        # self._widget.trajectory_list.setCurrentRow(self.moving_routines.keys().index(self.trajectory_selected))
        self.update_waypoints()

    def update_waypoints(self):
        self.reset_wp_markers()
        self._widget.waypoints_list.clear()
        if not self.trajectory_selected is None and self.trajectory_selected in self.moving_routines.keys():
            wps = ["x,\ty,\tz,\tyaw\tid".expandtabs(12)]
            for i,wp_pose in enumerate(self.moving_routines[self.trajectory_selected]):
                pose =ros_numpy.msgify(Pose,np.array(wp_pose))
                quarterion = [pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]
                ang = tf.transformations.euler_from_quaternion(quarterion)[2]
                x,y,z = pose.position.x, pose.position.y, pose.position.z
                wps+=["{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.0f}".format(x,y,z,ang,i+1).expandtabs(4)]

            self._widget.waypoints_list.addItems(wps)
            self.publish_wp_markers()

    def update_interface(self, event=None):
        """interface update loop"""

        # self.publish_drone_marker()
        stamp = os.stat(self.trajectories_path).st_mtime
        if stamp != self._cached_stamp:
            self._cached_stamp = stamp
            rospy.loginfo("Reloading trajectory file")
            rospy.sleep(1)
            self.load_moving_routines()
            rospy.loginfo("DONE reloading trajectory file")

        check_time_list = ["SLAM", "bebop odom","map odom"]

        min_dt = 0.2
        for k in check_time_list:
            if not self.checklist[k] is None and (rospy.Time.now() - self.checklist[k]).to_sec() > min_dt:
                self.checklist[k] = None

        ready_to_execute = True
        #change items color
        for i,(k,v) in enumerate(self.checklist.items()):
            self._widget.list.item(i).setBackground( QColor('#fdc086') if v is None else QColor('#7fc97f'))
            if v is None:
                ready_to_execute = False
        #enable and disable buttons
        if self.checklist["SLAM"] is None:
            self._widget.calibrate_origin.setEnabled(False)
            self._widget.calibrate_scale.setEnabled(False)
            self._widget.calibrate_angle.setEnabled(False)
            self._widget.save_map.setEnabled(False)

        else:
            self._widget.calibrate_origin.setEnabled(True)
            self._widget.calibrate_scale.setEnabled(True)
            self._widget.calibrate_angle.setEnabled(True)
            self._widget.save_map.setEnabled(True)
            
        
        if not ready_to_execute: #self.checklist["trajectory"] is None or self.checklist["calibrated"] is None:
            self._widget.execute.setEnabled(False)
        else:
            self._widget.execute.setEnabled(True)
        
        if self.trajectory_selected is None:
            self._widget.load.setEnabled(False)
        else:
            self._widget.load.setEnabled(True)

        if self.checklist["bebop odom"] is None:
            self._widget.takeoff.setEnabled(False)
            self._widget.land.setEnabled(False)
        else:
            self._widget.takeoff.setEnabled(True)
            self._widget.land.setEnabled(True)
        
        if self.checklist["map odom"] is None:
            self._widget.drone_pose.setText("no map odom")

        if self.checklist["map odom"] is None and self.trajectory_selected is None:
            self._widget.add_wp.setEnabled(False)
        else:
            self._widget.add_wp.setEnabled(True)
            
        if self.map_name in self.calibration_data.keys():
            if "origin_slam_pose" in self.calibration_data[self.map_name]:
                self._widget.calibrate_origin.setText("Recalibrate\norigin")
            else:
                self._widget.calibrate_origin.setText("Calibrate\norigin")
            if "scale_slam_pose" in self.calibration_data[self.map_name]:
                self._widget.calibrate_scale.setText("Recalibrate\nscale")
            else:
                self._widget.calibrate_scale.setText("Calibrate\nscale")
            if "scale_2_slam_pose" in self.calibration_data[self.map_name]:
                self._widget.calibrate_angle.setText("Recalibrate\nangle")
            else:
                self._widget.calibrate_angle.setText("Calibrate\nangle")
        else:
            self._widget.calibrate_origin.setText("Calibrate\norigin")
            self._widget.calibrate_scale.setText("Calibrate\nscale")
            self._widget.calibrate_angle.setText("Calibrate\nangle")
        
        if self.waypoint_selected_index is None:
            self._widget.del_wp.setEnabled(False)
            self._widget.edit_wp.setEnabled(False)

        else:
            self._widget.del_wp.setEnabled(True)
            self._widget.edit_wp.setEnabled(True)


        if self.manual_mode:
            self._widget.up.setEnabled(True)
            self._widget.down.setEnabled(True)
            self._widget.front.setEnabled(True)
            self._widget.back.setEnabled(True)
            self._widget.right.setEnabled(True)
            self._widget.left.setEnabled(True)
            self._widget.rotate_cw.setEnabled(True)
            self._widget.rotate_acw.setEnabled(True)
        else:
            self._widget.up.setEnabled(False)
            self._widget.down.setEnabled(False)
            self._widget.front.setEnabled(False)
            self._widget.back.setEnabled(False)
            self._widget.right.setEnabled(False)
            self._widget.left.setEnabled(False)
            self._widget.rotate_cw.setEnabled(False)
            self._widget.rotate_acw.setEnabled(False)

    def shutdown_plugin(self):
        # TODO unregister all publishers here

        self._odom_sub.unregister()
        self._slam_sub.unregister()
        self.current_pose_sub.unregister()

        print("shutdown_plugin")
        pass

    def save_settings(self, plugin_settings, instance_settings):
        # TODO save intrinsic configuration, usually using:
        # instance_settings.set_value(k, v)
        print("save_settings")
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        # TODO restore intrinsic configuration, usually using:
        # v = instance_settings.value(k)
        print("restore_settings")
        pass

    # def trigger_configuration(self):
    #     pass
        # Comment in to signal that the plugin has a way to configure
        # This will enable a setting button (gear icon) in each dock widget title bar
        # Usually used to open a modal configuration dialog