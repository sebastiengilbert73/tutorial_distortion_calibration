import copy
import os
import logging
import argparse
import cv2
import numpy as np
import torch
import math

class RadialDistortion():
    def __init__(self, image_sizeHW, center=(0.5, 0.5), alpha=0.0):
        self.center = center
        self.alpha = alpha
        self.image_sizeHW = image_sizeHW

    def GroupCheckerboardPoints(self, intersection_points_list, grid_shapeHW):
        if len(intersection_points_list) != grid_shapeHW[0] * grid_shapeHW[1]:
            raise ValueError(f"RadialDistortion.GroupCheckerboardPoints(): len(intersection_points_list) ({len(intersection_points_list)}) != grid_shapeHW[0] * grid_shapeHW[1] ({grid_shapeHW[0] * grid_shapeHW[1]})")

        number_of_horizontal_lines = grid_shapeHW[0]
        number_of_vertical_lines = grid_shapeHW[1]

        # Vertical lines
        p_sorted_by_y = sorted(intersection_points_list, key=lambda x: x[1])

        horizontal_lines = []
        for horizontal_line_ndx in range(number_of_horizontal_lines):
            start_ndx = horizontal_line_ndx * number_of_vertical_lines
            end_ndx = (horizontal_line_ndx + 1) * number_of_vertical_lines
            horizontal_line = p_sorted_by_y[start_ndx: end_ndx]
            horizontal_lines.append(horizontal_line)

        # Vertical lines
        p_sorted_by_x = sorted(intersection_points_list, key=lambda x: x[0])

        vertical_lines = []
        for vertical_line_ndx in range(number_of_vertical_lines):
            start_ndx = vertical_line_ndx * number_of_horizontal_lines
            end_ndx = (vertical_line_ndx + 1) * number_of_horizontal_lines
            vertical_line = p_sorted_by_x[start_ndx: end_ndx]
            vertical_lines.append(vertical_line)

        return horizontal_lines, vertical_lines

    def Optimize(self,
                 intersection_points_list,
                 grid_shapeHW,
                 learning_rate=0.1,
                 momentum=0.9,
                 weight_decay=0,
                 number_of_epochs=100):
        horizontal_lines, vertical_lines = self.GroupCheckerboardPoints(intersection_points_list, grid_shapeHW)
        line_points_tsr = torch.cat([torch.tensor(horizontal_lines), torch.tensor(vertical_lines)], dim=0)  # (N_line, n_pts_per_line, 2)

        neural_net = DistortionParametersOptimizer((self.image_sizeHW[1]//2, self.image_sizeHW[0]//2), 0.00000,
                                                   self.image_sizeHW)
        optimizer = torch.optim.Adam(neural_net.parameters(), lr=learning_rate, betas=(momentum, 0.999),
                                     weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()
        epoch_loss_center_alpha_list = []
        for epoch in range(1, number_of_epochs + 1):
            logging.info("*** Epoch {} ***".format(epoch))
            neural_net.train()

            target_output_tsr = torch.zeros(line_points_tsr.shape[0])
            optimizer.zero_grad()
            output_tsr = neural_net(line_points_tsr)
            loss = criterion(output_tsr, target_output_tsr)
            loss.backward()
            optimizer.step()
            print(f"loss.item() = {loss.item()}")
            epoch_loss_center_alpha_list.append((epoch, loss.item(),
                                            (neural_net.center[0].item() * self.image_sizeHW[1], neural_net.center[1].item() * self.image_sizeHW[0]),
                                            neural_net.alpha.item()))

        self.center = (neural_net.center[0].item() * self.image_sizeHW[1], neural_net.center[1].item() * self.image_sizeHW[0])
        self.alpha = neural_net.alpha.item()
        return epoch_loss_center_alpha_list

    def UndistortPoint(self, point, must_be_rounded=False):
        shifted_p = (point[0] - self.center[0], point[1] - self.center[1])
        shifted_p_scaled = (shifted_p[0]/self.image_sizeHW[1], shifted_p[1]/self.image_sizeHW[0])
        shifted_p_squared_sum = shifted_p_scaled[0]**2 + shifted_p_scaled[1]**2
        distorted_radius = math.sqrt(shifted_p_squared_sum)
        undistortion_factor = 1.0 + self.alpha * distorted_radius**2
        scaled_shifted_undistorted_p = (undistortion_factor * shifted_p_scaled[0], undistortion_factor * shifted_p_scaled[1])
        shifted_undistorted_p = (scaled_shifted_undistorted_p[0] * self.image_sizeHW[1], scaled_shifted_undistorted_p[1] * self.image_sizeHW[0])
        undistorted_p = (shifted_undistorted_p[0] + self.center[0], shifted_undistorted_p[1] + self.center[1])
        if must_be_rounded:
            return (round(undistorted_p[0]), round(undistorted_p[1]))
        else:
            return undistorted_p

    def UndistortImage(self, image, fill_with_local_median=False):
        undistorted_img = np.zeros(image.shape, dtype=np.uint8)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                undistorted_p = self.UndistortPoint((x, y), must_be_rounded=True)
                if undistorted_p[0] >= 0 and undistorted_p[0] < undistorted_img.shape[1] and \
                    undistorted_p[1] >= 0 and undistorted_p[1] < undistorted_img.shape[0]:
                    undistorted_img[undistorted_p[1], undistorted_p[0], :] = image[y, x]
        if fill_with_local_median:
            for y in range(undistorted_img.shape[0]):
                for x in range(undistorted_img.shape[1]):
                    if undistorted_img[y, x].all() == 0:

                        neighborhood_rect = [max(x - 2, 0), max(y - 2, 0), 5, 5]
                        neighborhood_img = undistorted_img[neighborhood_rect[1]: neighborhood_rect[1] + neighborhood_rect[3],
                                           neighborhood_rect[0]: neighborhood_rect[0] + neighborhood_rect[2], :]
                        for channel in range(3):
                            median = np.median(neighborhood_img[:, :, channel])
                            undistorted_img[y, x, channel] = median

        return undistorted_img


class DistortionParametersOptimizer(torch.nn.Module):
    def __init__(self, center, alpha, image_sizeHW, zero_threshold=1e-12):
        super(DistortionParametersOptimizer, self).__init__()
        self.center = torch.nn.Parameter(torch.tensor([center[0]/image_sizeHW[1], center[1]/image_sizeHW[0]]).float())
        self.alpha = torch.nn.Parameter(torch.tensor(alpha).float())
        self.image_sizeHW = image_sizeHW
        self.zero_threshold = zero_threshold

    def forward(self, input_tsr):  # input_tsr.shape = (N, n_points, 2)
        error_tsr = torch.zeros(input_tsr.shape[0])  # (N)
        scaled_input_tsr = self.ScaleWithDimensions(input_tsr)
        #shifted_points_tsr = input_tsr - self.center  # (N, n_points, 2)
        undistorted_points_tsr = self.UndistortTensor(scaled_input_tsr)  # (N, n_points, 2)
        for line_ndx in range(undistorted_points_tsr.shape[0]):
            line_error = torch.tensor(0).double()
            line_points_tsr = undistorted_points_tsr[line_ndx]  # (n_points, 2)
            line = self.Line(line_points_tsr)  # (rho, theta)
            #print(f"DistortionParametersOptimizer.forward(): line_ndx = {line_ndx}; line = {line}")
            for point_ndx in range(line_points_tsr.shape[0]):
                projection, distance = self.Project(line_points_tsr[point_ndx], line[0], line[1])
                error = torch.sum(torch.pow(projection - line_points_tsr[point_ndx], 2))
                line_error += error
            #print(f"DistortionParametersOptimizer.forward(): line_error = {line_error}")
            error_tsr[line_ndx] = line_error
        return error_tsr

    def Line(self, points_tsr):  # points_tsr.shape = (n_points, 2)
        xs = points_tsr[:, 0]
        min_x = min(xs)
        max_x = max(xs)
        if abs(min_x - max_x).item() <= self.zero_threshold:  # Vertical line
            return (min_x, torch.tensor(0))
        else:  # Non-vertical line
            ys = points_tsr[:, 1]
            min_y = min(ys)
            max_y = max(ys)
            if abs(min_y - max_y).item() <= self.zero_threshold:  # Horizontal line
                return (min_y, torch.tensor(torch.pi/2))
            else:  # Non-horizontal line
                # | x_i    y_i   -1  | | cos(theta) | = | 0 |
                # | ...    ...   ... | | sin(theta) | = | 0 |
                # | ...    ...   ... | |    rho     |   |...|
                A = torch.zeros((points_tsr.shape[0], 3))
                for row in range(points_tsr.shape[0]):
                    x_i = points_tsr[row][0]
                    y_i = points_tsr[row][1]
                    A[row, 0] = x_i
                    A[row, 1] = y_i
                    A[row, 2] = -1
                # Solution to a system of homogeneous linear equations. Cf. https://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy
                e_vals, e_vecs = torch.linalg.eig(torch.matmul(A.T, A))
                #print(f"DistortionParametersOptimizer.Line(): torch.matmul(A.T, A) = {torch.matmul(A.T, A)}")
                # Extract the eigenvector (column) associated with the minimum eigenvalue
                #print (f"DistortionParametersOptimizer.Line(): e_vals = {e_vals}")
                z = e_vecs[:, torch.argmin(e_vals.real)]
                # Multiply by a factor such that cos^2(theta) + sin^2(theta) = 1
                r2 = z[0] ** 2 + z[1] ** 2
                if abs(r2) < self.zero_threshold:
                    raise ValueError(f"DistortionParametersOptimizer.Line(): z[0]**2 + z[1]**2 ({r2}) < {self.zero_threshold}")
                z = z / torch.sqrt(r2)
                #print(f"DistortionParametersOptimizer.Line(): z = {z}")
                theta = torch.angle(torch.complex(z[0].real, z[1].real))
                rho = z[2].real
                return (rho, theta)

    def Project(self, p, rho, theta):
        # | sin(theta)      cos(theta) | | beta  | = |    x - rho * cos(theta)    |
        # | -cos(theta)     sin(theta) | | gamma |   |    y - rho * sin(theta)    |
        # beta is the distance between the central point (rho * cos(theta), rho * sin(theta)) and the projection
        # gamma is the distance between the point of interest and the projection
        A = torch.tensor([[math.sin(theta), math.cos(theta)], [-math.cos(theta), math.sin(theta)]])
        b = torch.tensor([p[0] - rho * math.cos(theta), p[1] - rho * math.sin(theta)])
        #print(f"DistortionParametersOptimizer.Project(): A.dtype = {A.dtype}; b.dtype = {b.dtype}")
        beta_gamma = torch.linalg.solve(A, b)
        p_prime = torch.tensor([rho * math.cos(theta) + beta_gamma[0] * math.sin(theta),
                               rho * math.sin(theta) - beta_gamma[0] * math.cos(theta)])
        return p_prime, beta_gamma[1]  # gamma is the distance between the point of interest and the projection

    def Undistort(self, p):
        shifted_p = p - self.center
        shifted_p_squared = torch.pow(shifted_p, 2)
        shifted_p_squared_sum = torch.sum(shifted_p_squared,dim=0)# torch.tensor(shifted_p_squared[0] + shifted_p_squared[1])#
        distorted_radius = torch.sqrt(shifted_p_squared_sum)
        undistortion_factor = torch.tensor(1.0) + self.alpha * torch.pow(distorted_radius, 2)
        shifted_undistorted_p = undistortion_factor * shifted_p
        unshifted_undistorted_p = shifted_undistorted_p + self.center
        return unshifted_undistorted_p

    def UndistortTensor(self, points_tsr):  # points_tsr.shape = (N, n_points, 2)
        undistorted_points_tsr = torch.zeros(points_tsr.shape)  # (N, n_points, 2)
        for line_ndx in range(points_tsr.shape[0]):
            line_pts_tsr = points_tsr[line_ndx]  # (n_points, 2)
            for pt_ndx in range(line_pts_tsr.shape[0]):
                undistorted_pt = self.Undistort(line_pts_tsr[pt_ndx])
                undistorted_points_tsr[line_ndx, pt_ndx, :] = undistorted_pt
        return undistorted_points_tsr

    def ScaleWithDimensions(self, input_tsr):  # (N, n_points, 2)
        scaled_tsr = torch.zeros(input_tsr.shape)  # (N, n_points, 2)
        for line_ndx in range(input_tsr.shape[0]):
            for pt_ndx in range(input_tsr.shape[1]):
                scaled_tsr[line_ndx, pt_ndx, 0] = input_tsr[line_ndx, pt_ndx, 0]/self.image_sizeHW[1]
                scaled_tsr[line_ndx, pt_ndx, 1] = input_tsr[line_ndx, pt_ndx, 1] / self.image_sizeHW[0]
        return scaled_tsr