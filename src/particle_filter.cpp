/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  // Set the number of particles
  num_particles = 100;

  // use a random number generator
  default_random_engine gen;

  // create a gaussian/normal distribution for x, y and theta
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // create particles and initialize x, y, theta based on normal distribution around the GPS measurement
  for (int i = 0; i < num_particles; i++) {
      Particle particle;
      particle.id = i;
      particle.x = dist_x(gen);
      particle.y = dist_y(gen);
      particle.theta = dist_theta(gen);
      particle.weight = 1.0;
      particles.push_back(particle);
      weights.push_back(1.0);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

   default_random_engine gen;
   normal_distribution<double> dist_x(0, std_pos[0]);
   normal_distribution<double> dist_y(0, std_pos[1]);
   normal_distribution<double> dist_theta(0, std_pos[2]);


   for (int i = 0; i < num_particles; i++) {
       if (fabs(yaw_rate) < 0.00001) {
           particles[i].x += velocity * delta_t * cos(particles[i].theta);
           particles[i].y += velocity * delta_t * sin(particles[i].theta);
       } else {
           particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
           particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
           particles[i].theta += yaw_rate * delta_t;
       }

       // add random gaussian noise
       particles[i].x += dist_x(gen);
       particles[i].y += dist_y(gen);
       particles[i].theta += dist_theta(gen);
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

   for (int i = 0; i < observations.size(); i++) {
       LandmarkObs o = observations[i];

       double min_dist = numeric_limits<double>::max();
       int map_id = -1;
       for (int j = 0; j < predicted.size(); j++) {
           LandmarkObs p = predicted[j];
           double cur_dist = dist(o.x, o.y, p.x, p.y);
           if (cur_dist < min_dist) {
               min_dist = cur_dist;
               map_id = p.id;
           }
       }
       observations[i].id = map_id;
   }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

   // for each particle
   for (int i = 0; i < num_particles; i++) {
       double p_x = particles[i].x;
       double p_y = particles[i].y;
       double p_theta = particles[i].theta;

       // find landmarks within sensor range of this particle
       vector<LandmarkObs> predictions;
       for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
           float landmark_x = map_landmarks.landmark_list[j].x_f;
           float landmark_y = map_landmarks.landmark_list[j].y_f;
           int landmark_id = map_landmarks.landmark_list[j].id_i;

           // choose landmark within sensor range
           if (fabs(landmark_x - p_x) <= sensor_range && fabs(landmark_y - p_y) <= sensor_range) {
               predictions.push_back(LandmarkObs {landmark_id, landmark_x, landmark_y});
           }
       }

       // transform car sensor landmark observations from the car coordinate system to the map coordinate system
       vector<LandmarkObs> transformed_observations;
       for (int j = 0; j < observations.size(); j++) {
           double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
           double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
           transformed_observations.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
       }

       // find which observations corresponding to which landmarks
       dataAssociation(predictions, transformed_observations);

       // reinitialize the weight of this particle
       particles[i].weight = 1.0;

       for (int j = 0; j < transformed_observations.size(); j++) {
           double observation_x, observation_y, predicted_x, predicted_y;
           observation_x = transformed_observations[j].x;
           observation_y = transformed_observations[j].y;

           int associated_prediction = transformed_observations[j].id;
           for (int k = 0; k < predictions.size(); k++) {
               if (predictions[k].id == associated_prediction) {
                   predicted_x = predictions[k].x;
                   predicted_y = predictions[k].y;
               }
           }

           // calculate weight for this particle using multivariate gaussian
           double s_x = std_landmark[0];
           double s_y = std_landmark[1];
           double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(predicted_x-observation_x,2)/(2*pow(s_x, 2)) + (pow(predicted_y-observation_y,2)/(2*pow(s_y, 2)))));

           particles[i].weight *= obs_w;
       }
   }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    vector<Particle> new_particles;
    default_random_engine gen;

    // get all of the current weights
    vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    uniform_int_distribution<int> uniintdist(0, num_particles-1);
    auto index = uniintdist(gen);

    double max_weight = *max_element(weights.begin(), weights.end());

    uniform_real_distribution<double> unirealdist(0.0, max_weight);

    double beta = 0.0;

    for (int i = 0; i < num_particles; i++) {
        beta += unirealdist(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }

    particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}