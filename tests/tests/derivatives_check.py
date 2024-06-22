import numpy as np
from scipy.spatial.transform import Rotation as R


class depthConsistencyErrorTerm:
    def __init__(self):
        self.dim = 10
        self.mask = np.asarray([0, 0, 1])

    def __call__(self, x: list) -> float:
        ########################
        # x[:3]  angle-axis    #
        # x[3:6] translation   #
        # x[6:9] landmark      #
        # x[9]   scale         #
        # from Camera to World #
        # i.e. w_var_c         #
        ########################
        assert len(x) == self.dim

        ######################################################
        # Canonical constraint                               #
        ######################################################
        # ||(c_R_w*w_l + c_t_w))z() - (c_S*c_C_j)||          #
        # Write as follows: need Jacobians from Cam -> World #
        # In optimization we assume canonical as measurement #
        ######################################################
        # ||c_C_j - {(c_R_w*w_l - (c_R_w * w_t_c))z()/c_S}|| #
        ######################################################
        # The below will be usefull for rotation Jacobian    #
        # ||c_C_j - {(w_R_c.T*(w_l - w_t_c))z()/c_S}||       #
        ######################################################

        # unpack #
        rot_angle_axis = np.asarray([x[0], x[1], x[2]])
        w_translation_c = np.asarray([x[3], x[4], x[5]])
        w_landmark = np.asarray([x[6], x[7], x[8]])
        scale = x[9]

        # rotation matrices #
        w_rotation_matrix_c = np.asarray(R.from_rotvec(rot_angle_axis).as_matrix())
        c_rotation_matrix_w = w_rotation_matrix_c.T

        # landmark from world to camera #
        c_landmark = (np.dot(c_rotation_matrix_w, w_landmark) - np.dot(c_rotation_matrix_w, w_translation_c))

        return (1.0 / scale) * np.dot(c_landmark, self.mask.T)

    def numerical_grad(self, x: list, h: float = 1e-6) -> np.ndarray:
        assert self.dim == len(x)

        x_local = x.copy()

        numerical_grad = np.zeros_like(x_local)

        # Translation, landmark, scale #
        for i in range(3, self.dim):
            mask_grad = np.zeros(self.dim)
            mask_grad[i] = 1

            f1 = self.__call__(x_local + h * mask_grad)
            f2 = self.__call__(x_local - h * mask_grad)

            numerical_grad[i] = (f1 - f2) / (2 * h)

        # Rotation is a manifold  #
        # Perturb around zero #
        rot_angle_axis = np.asarray([x_local[0], x_local[1], x_local[2]])
        w_rotation_matrix_c = np.asarray(R.from_rotvec(rot_angle_axis).as_matrix())
        c_rotation_matrix_w = w_rotation_matrix_c.T
        ###############
        #  0  -qz  ay #
        #  az  0  -qx #
        # -qy qx   0  #
        ###############

        #################################################
        # c_R_w = \delta(w_q) * \tilda{c_R_w}           #
        # w_R_c = c_R_w.T                               #
        # Perturb w_q. Then from w_R_c find the new w_q #
        #################################################

        # qx petrubation #
        dqx_plus = np.eye(3) - np.asarray([
            [0, 0, 0],
            [0, 0, -(0 + h)],
            [0, 0 + h, 0],
        ])
        c_R_w_new = dqx_plus @ c_rotation_matrix_w
        w_R_c_new = c_R_w_new.T
        w_q_c_new = R.from_matrix(w_R_c_new)
        x_local[0:3] = w_q_c_new.as_rotvec()
        f1 = self.__call__(x_local)

        dqx_minus = np.eye(3) - np.asarray([
            [0, 0, 0],
            [0, 0, -(0 - h)],
            [0, 0 - h, 0],
        ])
        c_R_w_new = dqx_minus @ c_rotation_matrix_w
        w_R_c_new = c_R_w_new.T
        w_q_c_new = R.from_matrix(w_R_c_new)
        x_local[0:3] = w_q_c_new.as_rotvec()
        f2 = self.__call__(x_local)

        # Grad qx #
        numerical_grad[0] = (f1 - f2) / (2 * h)

        # qy petrubation #
        dqy_plus = np.eye(3) - np.asarray([
            [0, 0, 0 + h],
            [0, 0, 0],
            [-(0 + h), 0, 0],
        ])
        c_R_w_new = dqy_plus @ c_rotation_matrix_w
        w_R_c_new = c_R_w_new.T
        w_q_c_new = R.from_matrix(w_R_c_new)
        x_local[0:3] = w_q_c_new.as_rotvec()
        f1 = self.__call__(x_local)

        dqy_minus = np.eye(3) - np.asarray([
            [0, 0, 0 - h],
            [0, 0, 0],
            [-(0 - h), 0, 0],
        ])
        c_R_w_new = dqy_minus @ c_rotation_matrix_w
        w_R_c_new = c_R_w_new.T
        w_q_c_new = R.from_matrix(w_R_c_new)
        x_local[0:3] = w_q_c_new.as_rotvec()
        f2 = self.__call__(x_local)

        # Grad qy #
        numerical_grad[1] = (f1 - f2) / (2 * h)

        # qz petrubation #
        dqz_plus = np.eye(3) - np.asarray([
            [0, -(0 + h), 0],
            [0 + h, 0, 0],
            [0, 0, 0],
        ])
        c_R_w_new = dqz_plus @ c_rotation_matrix_w
        w_R_c_new = c_R_w_new.T
        w_q_c_new = R.from_matrix(w_R_c_new)
        x_local[0:3] = w_q_c_new.as_rotvec()
        f1 = self.__call__(x_local)

        dqz_minus = np.eye(3) - np.asarray([
            [0, -(0 - h), 0],
            [0 - h, 0, 0],
            [0, 0, 0],
        ])
        c_R_w_new = dqz_minus @ c_rotation_matrix_w
        w_R_c_new = c_R_w_new.T
        w_q_c_new = R.from_matrix(w_R_c_new)
        x_local[0:3] = w_q_c_new.as_rotvec()
        f2 = self.__call__(x_local)

        # Grad qz #
        numerical_grad[2] = (f1 - f2) / (2 * h)

        return numerical_grad

    def analytical_grad(self, x: np.ndarray) -> np.ndarray:
        assert self.dim == len(x)
        ######################################################
        # Canonical constraint                               #
        ######################################################
        # ||(c_R_w*w_l + c_t_w))z() - (c_S*c_C_j)||          #
        # Write as follows: need Jacobians from Cam -> World #
        # In optimization we assume canonical as measurement #
        ######################################################
        # ||c_C_j - {(c_R_w*w_l - (c_R_w * w_t_c))z()/c_S}|| #
        ######################################################
        # The below will be usefull for rotation Jacobian    #
        # ||c_C_j - {(c_R_w*(w_l - w_t_c))z()/c_S}||         #
        ######################################################
        x_local = x.copy()
        analytical_grad = np.zeros_like(x_local)

        rot_angle_axis = np.asarray([x_local[0], x_local[1], x_local[2]])
        w_translation_c = np.asarray([x_local[3], x_local[4], x_local[5]])
        landmark_w = np.asarray([x_local[6], x_local[7], x_local[8]])
        scale = x_local[9]

        w_rotation_matrix_c = np.asarray(R.from_rotvec(rot_angle_axis).as_matrix())
        c_rotation_matrix_w = w_rotation_matrix_c.T

        ###############
        #  0  -qz  ay #
        #  az  0  -qx #
        # -qy qx   0  #
        #######################################################
        # q is in minimal repr                                #
        #######################################################
        # c_R_w = \delta(q) * \tilda{c_R_w}                   #
        # c_R_w * w_q_c = \delta(q) * \tilda{c_R_w} * w_q_c   #
        # (dc_R_W * w_q_c)/dw_q_c = \delta(q) * \tilda{c_R_w} #
        #######################################################
        delta_qx_dir = np.asarray([
            [0, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])

        delta_qy_dir = np.asarray([
            [0, 0, 1],
            [0, 0, 0],
            [-1, 0, 0],
        ])
        delta_qz_dir = np.asarray([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])
        ######################################################
        # The below will be usefull for rotation Jacobian    #
        # ||c_C_j - {(c_R_w*(w_l - w_t_c))z()/c_S}||         #
        ######################################################
        grad_c_R_w_wrt_qx = -delta_qx_dir @ c_rotation_matrix_w
        grad_c_R_w_wrt_qy = -delta_qy_dir @ c_rotation_matrix_w
        grad_c_R_w_wrt_qz = -delta_qz_dir @ c_rotation_matrix_w

        grad_canonical_wrt_q_x = np.dot(grad_c_R_w_wrt_qx, landmark_w - w_translation_c) / scale
        grad_canonical_wrt_q_y = np.dot(grad_c_R_w_wrt_qy, landmark_w - w_translation_c) / scale
        grad_canonical_wrt_q_z = np.dot(grad_c_R_w_wrt_qz, landmark_w - w_translation_c) / scale

        ##############
        # Final grad #
        # Apply mask #
        ##############
        analytical_grad[0] = np.dot(grad_canonical_wrt_q_x, self.mask)
        analytical_grad[1] = np.dot(grad_canonical_wrt_q_y, self.mask)
        analytical_grad[2] = np.dot(grad_canonical_wrt_q_z, self.mask)

        ######################################################
        # ||c_C_j - {(c_R_w*w_l - (c_R_w * w_t_c))z()/c_S}|| #
        ######################################################

        # Translation #
        analytical_grad[3] = -c_rotation_matrix_w[2][0] / scale
        analytical_grad[4] = -c_rotation_matrix_w[2][1] / scale
        analytical_grad[5] = -c_rotation_matrix_w[2][2] / scale

        # Landmark #
        analytical_grad[6] = c_rotation_matrix_w[2][0] / scale
        analytical_grad[7] = c_rotation_matrix_w[2][1] / scale
        analytical_grad[8] = c_rotation_matrix_w[2][2] / scale

        # Scale #
        landmark_c = (np.dot(c_rotation_matrix_w, landmark_w) - np.dot(c_rotation_matrix_w, w_translation_c))
        analytical_grad[9] = -np.dot(landmark_c, self.mask) / (scale**2)

        return analytical_grad


if __name__ == '__main__':
    ############################################
    # angle-axis, translation, landmark, scale #
    # from camera to world reference frame     #
    ############################################
    x = [0, np.pi / 2, 0, 0.2, 0.5, 0.6, 1.2, 2.2, 1.4, 3]

    dc = depthConsistencyErrorTerm()
    grad_numerical = dc.numerical_grad(x)
    grad_analytical = dc.analytical_grad(x)
    assert np.all(np.isclose(grad_numerical, grad_analytical, atol=0.00001))

    # Case 2
    x = [0, 0, np.pi / 2, 0.3, 1.5, 0.2, 0.2, 2.5, 1.2, 1.7]
    dc = depthConsistencyErrorTerm()
    grad_numerical = dc.numerical_grad(x)
    grad_analytical = dc.analytical_grad(x)
    assert np.all(np.isclose(grad_numerical, grad_analytical, atol=0.00001))

    # Case 3
    x = [np.pi / 2, 0, 0, 0.3, 1.5, 0.2, 0.2, 2.5, 1.2, 1.7]
    dc = depthConsistencyErrorTerm()
    grad_numerical = dc.numerical_grad(x)
    grad_analytical = dc.analytical_grad(x)

    assert np.all(np.isclose(grad_numerical, grad_analytical, atol=0.00001))

    # Case 4
    x = [np.pi / 3, 0, 0, 0.3, 1.5, 0.2, 0.2, 2.5, 1.2, 1.7]

    dc = depthConsistencyErrorTerm()
    grad_numerical = dc.numerical_grad(x)
    grad_analytical = dc.analytical_grad(x)
    assert np.all(np.isclose(grad_numerical, grad_analytical, atol=0.00001))

    # Case 5
    x = [0, np.pi / 1.5, 0, 0.3, 1.5, 0.2, 1.2, 2.4, 2.2, 4.7]
    dc = depthConsistencyErrorTerm()
    grad_numerical = dc.numerical_grad(x)
    grad_analytical = dc.analytical_grad(x)
    assert np.all(np.isclose(grad_numerical, grad_analytical, atol=0.00001))
