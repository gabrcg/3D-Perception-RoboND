#!/usr/bin/env python

# Import modules
import rospy
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

from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from pcl_helper import *


from rospy_message_converter import message_converter
import yaml



def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:
    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    # Note: this (1) is a poor choice of leaf size   
    # Experiment and find the appropriate size!
    LEAF_SIZE = 0.005 

    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

 
    

    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.5
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()

    # Assign axis and range to the passthrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.4
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()
    

    # RANSAC plane segmentation

    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance 
    # for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=True)


    # Much like the previous filters, we start by creating a filter object: 
    outlier_filter = extracted_inliers.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(10)
    # Set threshold scale factor
    x = .05
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # Finally call the filter function for magic
    extracted_inliers = outlier_filter.filter()

    # TODO: Euclidean Clustering
    
    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_inliers)# Apply function to convert XYZRGB to XYZ
    

    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameterscloud_filtered
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.015)
    ec.set_MinClusterSize(150)
    ec.set_MaxClusterSize(2000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    
    # TODO: Convert PCL data to ROS messages
    ros_objects = pcl_to_ros(cluster_cloud)
    #ros_objects = pcl_to_ros(extracted_inliers)
    #ros_table = pcl_to_ros(extracted_inliers)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_objects)
    

# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)

    # Classify the clusters!
    detected_objects_labels = []
    detected_objects = []
    labeled_features = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = extracted_inliers.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
        sample_cloud = pcl_to_ros(pcl_cluster)
        pcl_table_pub.publish(sample_cloud)

        # Extract histogram features
        # TODO: complete this step just as you did before in capture_features.py
        chists = compute_color_histograms(sample_cloud, using_hsv=True)
        normals = get_normals(sample_cloud)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        #labeled_features.append([feature, model_name])

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = sample_cloud
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)    


# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    test_scene_num = Int32()
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    dict_list = []

    labels = []
    centroids = [] # to be list of tuples (x, y, z)

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')

    # TODO: Parse parameters into individual variables
    object_list_group = []
    object_list_name = []
    dropbox_group = []

    for item in object_list_param:
        object_list_name.append(item['name'])
        object_list_group.append(item['group'])
    
    for item in dropbox_list_param:
        dropbox_group.append(item['group'])


    test_scene_num.data = 1


    # TODO: Loop through the pick list
    for object in object_list:
        
        #Fiding the object group and name
        object_group = object_list_param[object_list_name.index(object.label)]['group']
        object_name.data = str(object.label)


        #Fiding place_pose and arm name
        arm_name.data = str(dropbox_list_param[dropbox_group.index(object_group)]['name'])
        pose = (dropbox_list_param[dropbox_group.index(object_group)]['position'])
        place_pose.position = Point(pose[0],pose[1],pose[2])


        # Pick_pose
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroid = (np.mean(points_arr, axis=0)[:3])
        #asscalar solved TypeError: data type not understood
        pick_pose.position = Point(np.asscalar(centroid[0]),np.asscalar(centroid[1]),np.asscalar(centroid[2]))


        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

    yaml_filename = 'output_' + str(test_scene_num.data) + '.yaml'

    send_to_yaml(yaml_filename, dict_list)
        
    '''
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

        # TODO: Output your request parameters into output yaml file
    '''


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('object_recognition', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub= rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    
    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    

    # Initialize color_list    

    get_color_list.color_list = []
    '''
    try:
        pr2_mover()
    except rospy.ROSInterruptException:
        pass
    '''
    # TODO: Spin while node is not shutdown
        # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
