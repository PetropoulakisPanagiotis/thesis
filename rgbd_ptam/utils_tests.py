import g2o
import cv2
import numpy as np

from optimization import BundleAdjustmentScaleAware

def rpe(gt_pose, pose):
    gt_pose_mat = gt_pose.matrix()
    pose_mat = pose.matrix()

    # c2 -> c1: c1 -> c2
    # pose_mat: c2 -> c1 
    transformation_error = np.linalg.inv(gt_pose_mat) @ pose_mat
    
    return np.linalg.norm(transformation_error[:3, 3])


def get_bad_measurements(optimizer, args):
    bad_measurements = set()
    for edge in optimizer.active_edges():
        #print(edge.chi2())
        #print(edge.error())
        #print(edge.measurement())
        #print(edge.initial_estimate)
        #print(edge.id())
        #exit()
        if isinstance(edge, g2o.EdgeCustomCamera):
            if edge.chi2() > args.threshold_camera:
                bad_measurements.add(edge.id())

        if isinstance(edge, g2o.EdgeDepthConsistencyScale):
            print(edge.chi2())
            if edge.chi2() > args.threshold_depth_consistency:
                bad_measurements.add(edge.id())

        if isinstance(edge, g2o.EdgeScaleNetworkConsistency):
            if edge.chi2() > args.threshold_scale:
                bad_measurements.add(edge.id())

    bad_measurements = list(bad_measurements)
    return bad_measurements

def get_bad_measurements_simple(optimizer, args):
    bad_measurements = set()
    for edge in optimizer.active_edges():
        #print(edge.chi2())
        #exit()
        if isinstance(edge, g2o.EdgeCustomCamera):
            if edge.chi2() > args.threshold_camera:
                bad_measurements.add(edge.id())
                #print(edge.error())
                #print(edge.measurement())
                #print(edge.initial_estimate)
                #print(edge.id())
    bad_measurements = list(bad_measurements)
    return bad_measurements


def draw_keypoints(image, keypoints, name='keypoints', delay=0):
    if image.ndim == 2:
        image_copy = np.repeat(image[..., np.newaxis], 3, axis=2)
    else:
        image_copy = image
    image_copy = cv2.drawKeypoints(image_copy, keypoints, None, flags=0)
    cv2.imshow(name, image_copy)
    cv2.waitKey(delay)

def test_1(args, frame_1, matched_measurements_1, max_iterations=0, gt_pose=None):
    optimizer = BundleAdjustmentScaleAware(args=args)

    # Poses #
    optimizer.add_pose(0, frame_1.pose, frame_1.cam, fixed=False) 

    index_point = 0
    index_camera_edge = 0

    # Points frame 1 #
    for meas in matched_measurements_1:  
        pt = meas.mappoint

        xy = meas.xy 
        
        # 3D Landmark #
        optimizer.add_point(index_point, pt.position, fixed=False)
        optimizer.add_camera_edge(index_camera_edge, point_id=index_point, pose_id=0, meas=meas.xy)
    
        index_point += 1
        index_camera_edge += 1

    optimizer.optimize(max_iterations)

    print("gt pose:")
    print(gt_pose.matrix())

    print("Noisy pose before optimization:")
    print("RPE:", rpe(gt_pose, frame_1.pose))
    print(frame_1.pose.matrix())

    print("Measurements vs bad measurements")
    print(len(matched_measurements_1))
    print(len(get_bad_measurements_simple(optimizer, args)))

    print("Noisy pose after optimization:")
    pose_after_2 = optimizer.get_pose(0)
    print("RPE:", rpe(gt_pose, pose_after_2))
    print(pose_after_2.matrix())     


def test_2(args, frame_1, matched_measurements_1, class_id=0, max_iterations=0, gt_pose=None):
    optimizer = BundleAdjustmentScaleAware(args=args)

    # Poses #
    optimizer.add_pose(0, frame_1.pose, frame_1.cam, fixed=False) 

    # Scale #
    optimizer.add_scale(0, frame_1.scale_aware_frame.scales[class_id], fixed=False)

    # Scales edges network #
    information_1 = np.identity(1)
    if args.use_uncertainties:
        information_1 *= 1. / frame_1.scale_aware_frame.scales_uncertainty[class_id]
    optimizer.add_scale_edge(0, scale_id=3, meas=frame_1.scale_aware_frame.scales[class_id],
                                  information=information_1)

    index_point = 0
    index_camera_edge = 0
    index_scale = 30000

    # Points frame 1 #
    for meas in matched_measurements_1:  
        pt = meas.mappoint

        xy = meas.xy 
        
        # 3D Landmark #
        optimizer.add_point(index_point, pt.position, fixed=False)
        optimizer.add_camera_edge(index_camera_edge, point_id=index_point, pose_id=0, meas=meas.xy)
    
        information = np.identity(1)
        if args.use_uncertainties:
            information *= 1. / meas.covariance_canonical_measurement
        optimizer.add_depth_scale_consistency_edge(index_scale, point_id=index_point, pose_id=0, \
                                                   scale_id=0, meas=meas.canonical_measurement, information=information)

        index_point += 1
        index_camera_edge += 1
        index_scale += 1
    for edge in optimizer.edges():
        if isinstance(edge, g2o.EdgeScaleNetworkConsistency):
            #print(edge.error())
            #exit()
            pass
        if isinstance(edge, g2o.EdgeDepthConsistencyScale):
            pass
            #print(edge.initial_estimate)
            #print(edge.chi2())

    print("num edges: ", len(optimizer.edges()))
    optimizer.optimize(max_iterations)

    print("gt pose:")
    print(gt_pose.matrix())

    print("Noisy pose before optimization:")
    print("RPE:", rpe(gt_pose, frame_1.pose))
    print(frame_1.pose.matrix())

    print("Measurements vs bad measurements")
    print(len(matched_measurements_1))
    print(len(get_bad_measurements(optimizer, args)))

    print("Scales before:")
    print(frame_1.scale_aware_frame.scales[class_id])

    print("Noisy pose after optimization:")
    pose_after_2 = optimizer.get_pose(0)
    print("RPE:", rpe(gt_pose, pose_after_2))
    print(pose_after_2.matrix())     

    print("Scales after:")
    print(optimizer.get_scale(0))

def optimize(args, frame_1, frame_2, matched_measurements_1, matched_measurements_2, class_id=0, max_iterations=0, gt_pose=None):
    optimizer = BundleAdjustmentScaleAware(args=args)

    # Poses #
    #optimizer.add_pose(0, frame_1.pose, frame_1.cam, fixed=True)  # offset + 2 * 0 = offset 
    optimizer.add_pose(1, frame_1.pose, frame_2.cam, fixed=False) # offset + 2 = offset + 2

    """
    # Scales #
    optimizer.add_scale(2, frame_1.scale_aware_frame.scales[class_id], fixed=True)
    optimizer.add_scale(3, frame_2.scale_aware_frame.scales[class_id], fixed=True)


    # Scales edges network #
    information_1 = np.identity(1)
    information_2 = np.identity(1)
    if args.use_uncertainties:
        information_1 *= 1. / frame_1.scale_aware_frame.scales_uncertainty[class_id]
        information_2 *= 1. / frame_2.scale_aware_frame.scales_uncertainty[class_id]
    optimizer.add_scale_edge(4, scale_id=2, meas=frame_1.scale_aware_frame.scales[class_id],
                                  information=information_1)
    optimizer.add_scale_edge(5, scale_id=3, meas=frame_2.scale_aware_frame.scales[class_id],
                                  information=information_2)
    """

    index_point = 0
    index_general = 0
    points_start = 6
    depth_scale_consistenct_start = 10000
    camera_start = 50000

    # Points frame 1 #
    for meas in matched_measurements_1:  
        pt = meas.mappoint

        xy = meas.xy 
        # 3D Landmark #
        # offset + 2 * (index_point + 6) + 1
        # 0: offset + 12 + 1 
        optimizer.add_point(index_point + points_start, pt.position, fixed=True)
        #optimizer.add_camera_edge(index_general + camera_start, point_id=index_point + points_start, pose_id=0, meas=meas.xy)

       
        """ 
        information = np.identity(1)
        if args.use_uncertainties:
            information *= 1. / m.covariance_canonical_measurement
        optimizer.add_depth_scale_consistency_edge(index_general + depth_scale_consistenct_start, point_id=index_point + points_start, pose_id=0, \
                                                   scale_id=2, meas=meas.canonical_measurement, information=information)
        """ 
        index_general += 1
        index_point += 1
    
    # Frame 2 # 
    index_point = 0
    for meas in matched_measurements_2:  
        pt = meas.mappoint

        optimizer.add_camera_edge(index_general + camera_start, point_id=index_point + points_start, pose_id=1, meas=meas.xy)

        """
        information = np.identity(1)
        if args.use_uncertainties:
            information *= 1. / m.covariance_canonical_measurement
        optimizer.add_depth_scale_consistency_edge(index_general + depth_scale_consistenct_start, point_id=index_point + points_start, pose_id=1, \
                                                   scale_id=3, meas=meas.canonical_measurement, information=information)
        """
        index_general += 1
        index_point += 1
     
    optimizer.optimize(max_iterations)

    print("gt\n")
    print(gt_pose.matrix())

    print("Before (noise)\n")
    print("RPE: ", rpe(gt_pose, frame_2.pose))
    print(frame_2.pose.matrix())
    print("Scales\n")
    print(frame_1.scale_aware_frame.scales[class_id], frame_2.scale_aware_frame.scales[class_id])


    print(len(matched_measurements_2))
    print(len(get_bad_measurements(optimizer, args)))

    print("After\n")
    pose_after_2 = optimizer.get_pose(1)
    print("RPE: ", rpe(gt_pose, pose_after_2))
    print(pose_after_2.matrix())     

   

    #scale_1 = optimizer.get_scale(2)
    #scale_2 = optimizer.get_scale(3)

    #print("Scales\n")
    #print(scale_1, scale_2)

