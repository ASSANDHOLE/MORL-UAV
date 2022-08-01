#include <math.h>
#include <stdio.h>

double obs_dist_circle(double *obs_x, double *obs_y, double uav_x, double uav_y, double end_x, double end_y, double r,
                       int obs_cnt) {
    double delta_y = end_y - uav_y;
    double delta_x = end_x - uav_x;
    double line_length = pow(pow(delta_x, 2) + pow(delta_y, 2), 0.5);
    double cos = delta_x / line_length;
    double sin = delta_y / line_length;
    double tan = delta_y / delta_x;

    double dist = line_length;
    for (int i = 0; i < obs_cnt; i++) {
        double foot_x =
                (pow(delta_y, 2) * uav_x + pow(delta_x, 2) * obs_x[i] + (obs_y[i] - uav_y) * (delta_x) * (delta_y)) /
                pow(line_length, 2);
        double dist_to_line = fabs((obs_x[i] - foot_x) / sin);
        if (dist_to_line <= r) {
            double d = pow(pow(r, 2) - pow(dist_to_line, 2), 0.5);
            double intersect_x1 = foot_x - d * cos;
            double intersect_x2 = foot_x + d * cos;
            if ((intersect_x1 >= uav_x && intersect_x1 <= end_x) || (intersect_x1 <= uav_x && intersect_x2 >= end_x)) {
                double tmp_dist = fabs(intersect_x1 / cos);
                dist = fmin(tmp_dist, dist);
            }
            if ((intersect_x2 >= uav_x && intersect_x2 <= end_x) || (intersect_x2 <= uav_x && intersect_x2 >= end_x)) {
                double tmp_dist = fabs(intersect_x2 / cos);
                dist = fmin(tmp_dist, dist);
            }
        }
    }
    return dist;
}

int main() {
    double obs_x[3] = {6, 10, 11};
    double obs_y[3] = {4, 0, 10};
    double uav_x = 0, uav_y = 0, end_x = 10, end_y = 10, r = 2, obs_cnt = 3;
    double dist = obs_dist_circle(obs_x, obs_y, uav_x, uav_y, end_x, end_y, r, obs_cnt);
    printf("%f\n", dist);
}