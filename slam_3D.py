
import copy
import os
import numpy as np
import open3d
import pyquaternion
from viewer import Viewer
import cv2

class Key_frame(object):
    def __init__(self, id, point_cloud, pose):
        self.id = id
        self.pose = copy.deepcopy(pose)
        self.point_cloud = copy.deepcopy(point_cloud)
        self.transformed_cloud = copy.deepcopy(point_cloud)

        self.node = open3d.pipelines.registration.PoseGraphNode(pose)
        self.transform_cloud()

    def transform_cloud(self):
        self.transformed_cloud.points = self.point_cloud.points
        self.transformed_cloud.normals = self.point_cloud.normals
        self.transformed_cloud.transform(self.node.pose)

class GraphSLAM(object):
    def __init__(self, window = 1000):
        self.graph = open3d.pipelines.registration.PoseGraph()
        self.keyframes = []
        self.window = window
        self.T = np.eye(4)

        self.keyframe_angle_thresh_deg = 25
        self.keyframe_trans_thresh_m = 2.
        self.voxel_size = 0.2

        self.viewer = Viewer()

        self.optimization = open3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=1.0,
                                                              edge_prune_threshold=0.25, reference_node=0)
        self.criteria = open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)

    def update(self, cloud, img):
        cloud = cloud.voxel_down_sample(voxel_size=self.voxel_size)
        if not len(self.keyframes):
            self.keyframes.append(Key_frame(id=0, point_cloud=cloud, pose=self.T))
            if len(self.keyframes)>self.window:
                self.keyframes.pop(0)
            self.graph.nodes.append(self.keyframes[-1].node)
            self.viewer.update_pose(pose=self.keyframes[-1].pose)
            self.viewer.update_image(img)
            return

        if not self.update_keyframe(cloud, img):
            return

        print('optimizing...')
        print(open3d.pipelines.registration.global_optimization(self.graph,open3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                      open3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                      self.optimization))
        for keyframe in self.keyframes:
            keyframe.transform_cloud()

    def update_keyframe(self, cloud, img, skip=20):
        reg = open3d.pipelines.registration.registration_icp(cloud, self.keyframes[-1].point_cloud, 1.0,
                                                   self.T, open3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                   criteria=self.criteria)

        angle = pyquaternion.Quaternion(matrix=reg.transformation[:3, :3]).degrees
        trans = np.linalg.norm(reg.transformation[:3, 3])

        if abs(angle) < self.keyframe_angle_thresh_deg and abs(trans) < self.keyframe_trans_thresh_m:
            self.T = reg.transformation
            return False

        pose = np.dot(self.keyframes[-1].pose, reg.transformation)
        self.keyframes.append(Key_frame(id=len(self.keyframes), point_cloud = cloud, pose = pose))
        if len(self.keyframes) > self.window:
            self.keyframes.pop(0)
        self.graph.nodes.append(self.keyframes[-1].node)
        self.T = np.eye(4)

        information = open3d.pipelines.registration.get_information_matrix_from_point_clouds(source=self.keyframes[-1].point_cloud,
                                                                                   target=self.keyframes[-2].point_cloud,
                                                                                   max_correspondence_distance=1.0,
                                                                                   transformation=reg.transformation)
        edge = open3d.pipelines.registration.PoseGraphEdge(self.keyframes[-1].id,self.keyframes[-2].id,
                                                 reg.transformation,information,uncertain=False)
        self.graph.edges.append(edge)

        cloud = np.asarray(self.keyframes[-1].transformed_cloud.points)[::skip]
        distance = np.linalg.norm(cloud, axis=1)
        MIN_DISTANCE, MAX_DISTANCE = np.min(distance), np.max(distance)
        colours = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
        colours = np.asarray([np.asarray(self.hsv_to_rgb(c, np.sqrt(1), 1.0)) for c in colours]) #[::-1]/np.max(colors)

        self.viewer.update_pose(pose=pose, cloud=cloud, colour=colours)
        self.viewer.update_image(img)

        return True

    def generate_map(self):
        map_cloud = open3d.geometry.PointCloud()
        for keyframe in self.keyframes:
            transformed = copy.deepcopy(keyframe.point_cloud)
            transformed.transform(keyframe.node.pose)
            map_cloud += transformed
        return map_cloud.voxel_down_sample(voxel_size=0.05)

    def hsv_to_rgb(self, h, s, v):
        if s == 0.0:
            return v, v, v

        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        if i == 0:
            return v, t, p
        if i == 1:
            return q, v, p
        if i == 2:
            return p, v, t
        if i == 3:
            return p, q, v
        if i == 4:
            return t, p, v
        if i == 5:
            return v, p, q

if __name__ == '__main__':
    dataset_path = '/home/eugeniu/Desktop/KITTI/scenario1'
    cloud_files = sorted([dataset_path + '/' + x for x in os.listdir(dataset_path) if '.pcd' in x])
    image_files = sorted([dataset_path + '/' + x for x in os.listdir(dataset_path) if '.png' in x])

    slam_obj = GraphSLAM()
    for i, cloud_file in enumerate(cloud_files[550:1700]):
        cloud = open3d.io.read_point_cloud(cloud_file)
        try:
            slam_obj.update(cloud, img=cv2.imread(image_files[i]))
        except Exception as err:
            print(err)
        print(i)

    open3d.visualization.draw_geometries([slam_obj.generate_map(),])
