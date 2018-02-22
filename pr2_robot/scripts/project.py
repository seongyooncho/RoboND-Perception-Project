#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)

    # Statistical Outlier Filtering
    outlier_filter = pcl_data.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(10)
    x = 0.001
    outlier_filter.set_std_dev_mul_thresh(x)
    cloud_filtered = outlier_filter.filter()

    # Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_downsampled = vox.filter()

    # PassThrough Filter
    passthrough = cloud_downsampled.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_passthroughed = passthrough.filter()

    passthrough = cloud_passthroughed.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.45
    axis_max = 0.45
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_passthroughed = passthrough.filter()

    passthrough = cloud_passthroughed.make_passthrough_filter()
    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.4
    axis_max = 0.9
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_passthroughed = passthrough.filter()

    # RANSAC Plane Segmentation
    seg = cloud_passthroughed.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    cloud_table = cloud_passthroughed.extract(inliers, negative=False)
    cloud_objects = cloud_passthroughed.extract(inliers, negative=True)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum clustering parameters
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(25000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
      for i, indices in enumerate(indices):
        color_cluster_point_list.append([white_cloud[indices][0],
                                         white_cloud[indices][1],
                                         white_cloud[indices][2],
                                         rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_filtered = pcl_to_ros(cloud_filtered)
    ros_cloud_downsampled = pcl_to_ros(cloud_downsampled)
    ros_cloud_passthroughed = pcl_to_ros(cloud_passthroughed)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_filtered_pub.publish(ros_cloud_filtered)
    pcl_downsampled_pub.publish(ros_cloud_downsampled)
    pcl_passthroughed_pub.publish(ros_cloud_passthroughed)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        # convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Compute the associated feature vector

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .1
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    dict_list = []

    labels = []
    centroids = []

    for detected_object in object_list:
        labels.append(detected_object.label)
        points_arr = ros_to_pcl(detected_object.cloud).to_array()
        np_means = np.mean(points_arr, axis=0)[:3]
        # Convert numpy.float64 to python float
        means = []
        for np_mean in np_means:
            means.append(np.asscalar(np_mean))
        centroids.append(means)

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
    test_scene_num_param = rospy.get_param('/test_scene_num')

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for i in range(len(object_list_param)):
        test_scene_num = Int32()
        object_name = String()
        arm_name = String()
        pick_pose = Pose()
        place_pose = Pose()

        test_scene_num.data = test_scene_num_param

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        object_name.data = object_list_param[i]['name']

        try:
            centroid = centroids[labels.index(object_name.data)]
            pick_pose.position.x = centroid[0]
            pick_pose.position.y = centroid[1]
            pick_pose.position.z = centroid[2]
        except ValueError:
            continue

        # TODO: Create 'place_pose' for the object
        # Find dropbox in dropbox_param that matches group
        dropbox = next((x for x in dropbox_param if x['group'] == object_list_param[i]['group']), None)
        if dropbox == None:
            print "Invalid object group: %s" % object_list_param[i]
            continue
        place_pose.position.x = dropbox['position'][0]
        place_pose.position.y = dropbox['position'][1]
        place_pose.position.z = dropbox['position'][2]

        # TODO: Assign the arm to be used for pick_place
        arm_name.data = dropbox['name']

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

        '''
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        '''

    # TODO: Output your request parameters into output yaml file
    send_to_yaml("output_%d.yaml"%test_scene_num.data, dict_list)




if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('object_recognition', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2,
                               pcl_callback, queue_size=1)

    # Create Publishers
    pcl_filtered_pub = rospy.Publisher("/pcl_filtered", PointCloud2, queue_size=1)
    pcl_downsampled_pub = rospy.Publisher("/pcl_downsampled", PointCloud2, queue_size=1)
    pcl_passthroughed_pub = rospy.Publisher("/pcl_passthroughed", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
