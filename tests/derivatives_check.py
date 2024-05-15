import numpy as np
from scipy.spatial.transform import Rotation as R


class depthConsistencyErrorTerm:

    def __init__(self):
        self.dim = 10
        self.mask = np.asarray([0, 0, 1])

    def __call__(self, x):
        # angle-axis, translation, landmark, scale #
        # all are from camera to world             #
        rot_angle_axis = np.asarray([x[0], x[1], x[2]])
        w_translation_c = np.asarray([x[3], x[4], x[5]])
        landmark_w = np.asarray([x[6], x[7], x[8]])
        scale = x[9]

        w_rotation_matrix_c = np.asarray(R.from_rotvec(rot_angle_axis).as_matrix())
        c_rotation_matrix_w = w_rotation_matrix_c.T

        landmark_c = (np.dot(c_rotation_matrix_w, landmark_w) - np.dot(c_rotation_matrix_w, w_translation_c))

        return (1.0 / scale) * np.dot(landmark_c, self.mask.T)

    def numerical_grad(self, x, h=1e-6):
        assert self.dim == len(x)

        grad = np.zeros_like(x)

        # Translation, landmark, scale #
        for i in range(3, self.dim):
            mask_grad = np.zeros(self.dim)
            mask_grad[i] = 1

            f1 = self.__call__(x + h*mask_grad)
            f2 = self.__call__(x - h*mask_grad)

            grad[i] = (f1 - f2) / (2 * h)
        print(grad)
        # Rotation is manifold - petrubation around zero #
        rot_angle_axis = np.asarray([x[0], x[1], x[2]])
        w_rotation_matrix_c = np.asarray(R.from_rotvec(rot_angle_axis).as_matrix())

        #x#
        dx_plus = np.eye(3) - np.asarray([
                                        [0, 0, 0],
                                        [0, 0, -(0 + h)],
                                        [0, 0 + h, 0],
                                        ])


        r_new = dx_plus @ w_rotation_matrix_c
        r_new = r_new.T
        r_new = R.from_matrix(r_new)

        x[0:3] = r_new.as_rotvec()

        f1 = self.__call__(x)

        dx_minus = np.eye(3) - np.asarray([
                                        [0, 0, 0],
                                        [0, 0, -(0 - h)],
                                        [0, 0 - h, 0],
                                        ])


        r_new = dx_minus @ w_rotation_matrix_c
        r_new = r_new.T
        r_new = R.from_matrix(r_new)

        x[0:3] = r_new.as_rotvec()

        f2 = self.__call__(x)
        grad[0] = (f1 - f2) / (2 * h)

        #y#
        dy_plus = np.eye(3) - np.asarray([
                                        [0, 0, 0 + h],
                                        [0, 0, 0],
                                        [-(0 + h), 0, 0],
                                        ])

        r_update = dy_plus @ w_rotation_matrix_c
        r_update = r_update.T
        new_r_x = R.from_matrix(r_update)
        x[0:3] = new_r_x.as_rotvec()

        f1 = self.__call__(x)

        dy_minus = np.eye(3) - np.asarray([
                                        [0, 0, 0 - h],
                                        [0, 0, 0],
                                        [-(0 - h), 0, 0],
                                        ])


        r_update = dy_minus @ w_rotation_matrix_c
        r_update = r_update.T
        new_r_x = R.from_matrix(r_update)
        x[0:3] = new_r_x.as_rotvec()

        f2 = self.__call__(x)
        grad[1] = (f1 - f2) / (2 * h)

        #z#
        dz_plus = np.eye(3) - np.asarray([
                                        [0, -(0 + h), 0],
                                        [0 + h, 0, 0],
                                        [0, 0, 0],
                                        ])


        r_update = dz_plus @ w_rotation_matrix_c
        r_update = r_update.T
        new_r_x = R.from_matrix(r_update)
        x[0:3] = new_r_x.as_rotvec()

        f1 = self.__call__(x)
        dz_minus = np.eye(3) - np.asarray([
                                        [0, -(0 - h), 0],
                                        [0 - h, 0, 0],
                                        [0, 0, 0],
                                        ])


        r_update = dz_minus @ w_rotation_matrix_c
        r_update = r_update.T
        new_r_x = R.from_matrix(r_update)
        x[0:3] = new_r_x.as_rotvec()

        f2 = self.__call__(x)
        grad[2] = (f1 - f2) / (2 * h)

        print(grad)

        return grad


    def analytical_grad(self, x):
        assert self.dim == len(x)

        grad = np.zeros_like(x)

        rot_angle_axis = np.asarray([x[0], x[1], x[2]])
        w_translation_c = np.asarray([x[3], x[4], x[5]])
        landmark_w = np.asarray([x[6], x[7], x[8]])
        scale = x[9]

        w_rotation_matrix_c = np.asarray(R.from_rotvec(rot_angle_axis).as_matrix())
        c_rotation_matrix_w = w_rotation_matrix_c.T

        landmark_c = (np.dot(c_rotation_matrix_w, landmark_w) - np.dot(c_rotation_matrix_w, w_translation_c))

        # Rotation angle-axis #
        w_angle_axis_cross_product_c = np.asarray([
                                                [0, -x[2], x[1]],
                                                [x[2], 0, -x[0]],
                                                [-x[1], x[0], 0],
                                              ])

        w_angle_axis_cross_product_c_grad_x = np.asarray([
                                                [0, 0, 0],
                                                [0, 0, -1],
                                                [0, 1, 0],
                                              ])

        w_angle_axis_cross_product_c_grad_y = np.asarray([
                                                [0, 0, 1],
                                                [0, 0, 0],
                                                [-1, 0, 0],
                                              ])
        w_angle_axis_cross_product_c_grad_z = np.asarray([
                                                [0, -1, 0],
                                                [1, 0, 0],
                                                [0, 0, 0],
                                              ])


        rotation_grad_x_angle_axis = -w_angle_axis_cross_product_c_grad_x @ c_rotation_matrix_w
        rotation_grad_y_angle_axis = -w_angle_axis_cross_product_c_grad_y @ c_rotation_matrix_w
        rotation_grad_z_angle_axis = -w_angle_axis_cross_product_c_grad_z @ c_rotation_matrix_w

        rotation_grad_x_angle_axis = np.dot(rotation_grad_x_angle_axis, landmark_w - w_translation_c)
        rotation_grad_y_angle_axis = np.dot(rotation_grad_y_angle_axis, landmark_w - w_translation_c)
        rotation_grad_z_angle_axis = np.dot(rotation_grad_z_angle_axis, landmark_w - w_translation_c)

        # Final grad # 
        grad[0] = np.dot(rotation_grad_x_angle_axis, self.mask) / scale 
        grad[1] = np.dot(rotation_grad_y_angle_axis, self.mask) / scale
        grad[2] = np.dot(rotation_grad_z_angle_axis, self.mask) / scale

        # Translation #
        grad[3] = -c_rotation_matrix_w[2][0] / scale
        grad[4] = -c_rotation_matrix_w[2][1] / scale
        grad[5] = -c_rotation_matrix_w[2][2] / scale

        # Landmark #
        grad[6] = c_rotation_matrix_w[2][0] / scale
        grad[7] = c_rotation_matrix_w[2][1] / scale
        grad[8] = c_rotation_matrix_w[2][2] / scale


        # Scale #
        grad[-1] = -np.dot(landmark_c, self.mask)/(scale ** 2)

        return grad

if __name__ == '__main__':
    # angle-axis, translation, landmark, scale #
    # all are from camera to world             #
    x = [0, np.pi/2, 0, 0.2, 0.5, 0.6, 1.2, 2.2, 1.4, 3]

    dc = depthConsistencyErrorTerm()

    grad_numerical = dc.numerical_grad(x)
    grad_analytical = dc.analytical_grad(x)

    if np.all(np.isclose(grad_numerical[:3], grad_analytical[:3], atol=0.00001)):
        print("rotations equal\n")
    else:
        print("Grad_numerical: ", grad_numerical[:3])
        print("Grad_analytical: ", grad_analytical[:3])
    if np.all(np.isclose(grad_numerical[3:6], grad_analytical[3:6], atol=0.00001)):
        print("translation equal\n")
    if np.all(np.isclose(grad_numerical[6:9], grad_analytical[6:9], atol=0.00001)):
        print("landmark equal\n")
    if np.all(np.isclose(grad_numerical[9], grad_analytical[9], atol=0.00001)):
        print("scale\n")

    # Case 2
    x = [0, 0, np.pi/2, 0.3, 1.5, 0.2, 0.2, 2.5, 1.2, 1.7]

    dc = depthConsistencyErrorTerm()

    grad_numerical = dc.numerical_grad(x)
    grad_analytical = dc.analytical_grad(x)

    if np.all(np.isclose(grad_numerical[:3], grad_analytical[:3], atol=0.00001)):
        print("rotations equal\n")
    else:
        print("Grad_numerical: ", grad_numerical[:3])
        print("Grad_analytical: ", grad_analytical[:3])
    if np.all(np.isclose(grad_numerical[3:6], grad_analytical[3:6], atol=0.00001)):
        print("translation equal\n")
    if np.all(np.isclose(grad_numerical[6:9], grad_analytical[6:9], atol=0.00001)):
        print("landmark equal\n")
    if np.all(np.isclose(grad_numerical[9], grad_analytical[9], atol=0.00001)):
        print("scale\n")

    # Case 3
    x = [np.pi/2, 0, 0, 0.3, 1.5, 0.2, 0.2, 2.5, 1.2, 1.7]

    dc = depthConsistencyErrorTerm()

    grad_numerical = dc.numerical_grad(x)
    grad_analytical = dc.analytical_grad(x)

    if np.all(np.isclose(grad_numerical[:3], grad_analytical[:3], atol=0.00001)):
        print("rotations equal\n")
    else:
        print("Grad_numerical: ", grad_numerical[:3])
        print("Grad_analytical: ", grad_analytical[:3])
    if np.all(np.isclose(grad_numerical[3:6], grad_analytical[3:6], atol=0.00001)):
        print("translation equal\n")
    if np.all(np.isclose(grad_numerical[6:9], grad_analytical[6:9], atol=0.00001)):
        print("landmark equal\n")
    if np.all(np.isclose(grad_numerical[9], grad_analytical[9], atol=0.00001)):
        print("scale\n")


    # Case 4
    x = [np.pi/3, 0, 0, 0.3, 1.5, 0.2, 0.2, 2.5, 1.2, 1.7]

    dc = depthConsistencyErrorTerm()

    grad_numerical = dc.numerical_grad(x)
    grad_analytical = dc.analytical_grad(x)

    if np.all(np.isclose(grad_numerical[:3], grad_analytical[:3], atol=0.00001)):
        print("rotations equal\n")
    else:
        print("Grad_numerical: ", grad_numerical[:3])
        print("Grad_analytical: ", grad_analytical[:3])
    if np.all(np.isclose(grad_numerical[3:6], grad_analytical[3:6], atol=0.00001)):
        print("translation equal\n")
    if np.all(np.isclose(grad_numerical[6:9], grad_analytical[6:9], atol=0.00001)):
        print("landmark equal\n")
    if np.all(np.isclose(grad_numerical[9], grad_analytical[9], atol=0.00001)):
        print("scale\n")

    # Case 5
    x = [0, np.pi/3, 0, 0.3, 1.5, 0.2, 0.2, 2.5, 1.2, 1.7]

    dc = depthConsistencyErrorTerm()

    grad_numerical = dc.numerical_grad(x)
    grad_analytical = dc.analytical_grad(x)

    if np.all(np.isclose(grad_numerical[:3], grad_analytical[:3], atol=0.00001)):
        print("rotations equal\n")
    else:
        print("Grad_numerical: ", grad_numerical[:3])
        print("Grad_analytical: ", grad_analytical[:3])
    if np.all(np.isclose(grad_numerical[3:6], grad_analytical[3:6], atol=0.00001)):
        print("translation equal\n")
    if np.all(np.isclose(grad_numerical[6:9], grad_analytical[6:9], atol=0.00001)):
        print("landmark equal\n")
    if np.all(np.isclose(grad_numerical[9], grad_analytical[9], atol=0.00001)):
        print("scale\n")
