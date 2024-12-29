
import numpy as np
import math

class Sheepdog:

    def __init__(self, id, sensing_range, pos, goal, sheep_num, movement_delta):
        self.id = id
        self.sensing_range = sensing_range
        self.position = pos
        self.goal = goal
        self.connections = []
        self.neighbor_LCM = []
        self.sheep_connections = []
        self.stray_sheep = []
        self.sheep_number = sheep_num
        self.LCM = np.array([0,0])
        self.previous_LCM = np.array([0,0])
        self.GCM = np.array([0,0])
        self.driving_distance = 2 * math.sqrt(self.sheep_number)
        self.movement_delta = movement_delta
        self.heading = np.array([0.0, 0.0])
        self.previous_heading = np.array([0,0])


    def __repr__(self):
        return self.position
    
    #Connections need to be restablished when dogs get out of sensing range
    # Add agents as its neighbors
    def connect(self, agents):
        self.connections = []
        for aa in agents:
            distance_to_dog = np.linalg.norm(self.position - aa.position)
            if(not(aa.id == self.id) and distance_to_dog <= self.sensing_range):
                self.connections.append(aa)
    
    # def connect(self, agents):
    #     for aa in agents:
    #         self.connections.append(aa)


    # Returns its neighbor set
    def neighbors(self):
        return self.connections
    
    # Returns its local sheep set
    def sheep_neighbors(self, sheep):
        self.sheep_connections = []
        for i in sheep:
            distance_to_sheep = np.linalg.norm(self.position - i.position)
            if(distance_to_sheep <= self.sensing_range):
                self.sheep_connections.append(i)
        return self.sheep_connections

    # Send its values to its neighboring sheepdogs
    def propagate(self):
        for neigh in self.neighbors():
            neigh.receive((self.LCM,len(self.sheep_connections)))
        return self.position

    # Receive values from its neighbors
    def receive(self, value):
        self.neighbor_LCM.append(value)

    # Generates the local center of mass of sheep
    def calculate_LCM(self):
        # Triggers flag the first time the function is run to 
        # ensure previous LCM is non zero
        indicator = 0
        if(self.previous_LCM[0] == 0 and self.previous_LCM[1] == 0):
            indicator = 1

        self.previous_LCM = np.copy(self.LCM)

        x_avg = 0
        y_avg = 0
        for i in self.sheep_connections:
            x_avg += i.position[0]
            y_avg += i.position[1]

        if(len(self.sheep_connections) > 0):
            x_avg /= (len(self.sheep_connections))
            y_avg /= (len(self.sheep_connections))
            
            self.LCM = np.copy(np.array([x_avg,y_avg]))
        
        if(indicator == 1):
            self.previous_LCM = np.copy(self.LCM)
            indicator = 0
        return
    
    #Generates GCM approximate from LCM of neighbors
    #consensus ui at time step t = zi + sum( aw(i,j)(xj-xi))
    #zi = LCM - previous_LCM
    #Unweighted approach
    #if sheepdogs are neighbors, aw(i,j) = 1, 0 otherwise
    #Weighted approach
    #aw(i,j) = num of sheep used for measurement / total number of sheep
    def calculate_GCM(self):
        self.calculate_LCM()
        if(self.GCM[0] == 0 and self.GCM[1] == 0):
            self.GCM = np.copy(self.LCM)
        zi = self.LCM - self.previous_LCM
        self.GCM += zi
        for i in self.neighbor_LCM:
            self.GCM += i - self.LCM
        
    def get_GCM(self):
        return self.GCM

    #Populates a list of all sheep in range that are outside of radius from GCM
    def sheep_outside_of_radius(self):
        self.stray_sheep = []

        self.calculate_GCM()
        for i in self.sheep_connections:
            distance_of_sheep = np.linalg.norm(self.GCM - i.position)
            if(distance_of_sheep >= self.driving_distance):
                self.stray_sheep.append((distance_of_sheep,i))
        
        self.stray_sheep.sort(key= lambda t: t[0])
        self.stray_sheep = [point[1] for point in self.stray_sheep]

        # print(self.stray_sheep)

        return self.stray_sheep

    #Move Sheepdog according to consensus dynamics and goal position
    #Look through list of sheep that are outside of 2*sqrt(N) radius of GCM
    #Select sheep that is closest to sheepdog and not closer to other sheepdogs
    #If list is empty, or other sheepdogs are closer, begin driving the flock
    def update_position(self):
        rescue_sheep = np.array([0.0, 0.0])
        self.previous_heading = np.copy(self.heading)
        flag = 0
        for i in self.sheep_outside_of_radius():
            flag = 0
            for j in self.connections:
                #distance of other sheepdogs to current sheep
                other_sheepdog_dist = np.linalg.norm(i.position - j.position)
                this_sheepdog_dist = np.linalg.norm(i.position - self.position)
                if(other_sheepdog_dist < this_sheepdog_dist):
                    flag = 1
            if(flag == 0):
                rescue_sheep = i
                break
        
        # Go to driving position if there are no stray sheep
        # or if all stray sheep have another sheepdog closer
        if(flag == 1 or len(self.stray_sheep) == 0):
            # print("driving time")
            driving_offset = self.GCM - self.goal
            driving_offset = driving_offset / np.linalg.norm(driving_offset) 
            driving_offset *= self.driving_distance * 2

            # Check if either sheepdog coordinate is closer to the goal than the estimated GCM
            # If so, move outwards and circle around to the back of the flock center
            sheepdog_x_distance_to_goal = abs(self.position[0] - self.goal[0])
            sheepdog_y_distance_to_goal = abs(self.position[1] - self.goal[1])
            gcm_x_distance_to_goal = abs(self.GCM[0] - self.goal[0])
            gcm_y_distance_to_goal = abs(self.GCM[1] - self.goal[1])

            if(sheepdog_x_distance_to_goal <= gcm_x_distance_to_goal or sheepdog_y_distance_to_goal <= gcm_y_distance_to_goal):
                print("Driving before:", driving_offset)
                driving_offset = self.rotate(driving_offset, "drive", rescue_sheep)
                print("Driving after:", driving_offset)
            
            driving_position = self.GCM - driving_offset

        # Go to collecting position otherwise
        else:
            # print("collecting time")
            collecting_offset = rescue_sheep.position - self.GCM
            collecting_offset = collecting_offset / np.linalg.norm(collecting_offset)
            collecting_offset *= 2

            # Check if either sheepdog coordinate is closer to the goal than the estimated GCM
            # If so, move outwards and circle around to the back of the flock center
            sheepdog_x_distance_to_gcm = abs(self.position[0] - self.GCM[0])
            sheepdog_y_distance_to_gcm = abs(self.position[1] - self.GCM[1])
            sheep_x_distance_to_gcm = abs(rescue_sheep.position[0] - self.GCM[0])
            sheep_y_distance_to_gcm = abs(rescue_sheep.position[1] - self.GCM[1])

            if(sheepdog_x_distance_to_gcm <= sheep_x_distance_to_gcm or sheepdog_y_distance_to_gcm <= sheep_y_distance_to_gcm):
                print("Collecting before:", collecting_offset)
                collecting_offset = self.rotate(collecting_offset, "collect", rescue_sheep)
                print("Collecting after:", collecting_offset)

            driving_position = rescue_sheep.position - collecting_offset

        self.heading = driving_position - self.position
        self.heading = self.heading / np.linalg.norm(self.heading)

        #If the sheepdog is within 3 times the repulsion radius of any sheep, it stops moving
        for i in self.sheep_connections:
            distance_of_sheep = np.linalg.norm(self.position - i.position)
            # print(f"Distance to sheep:{distance_of_sheep}, repel range: {i.repulsion_range}")

            if(distance_of_sheep <= 3 * i.repulsion_range):
                self.heading = self.heading = np.array([0.0, 0.0])

        self.position += self.movement_delta * self.heading 



    # Calculate angle between either gcm, sheepdog, and goal, OR
    # Rescue sheep, sheepdog, and gcm depending on the angle type
    # Use rotation matrix on calculated offset:
        # [Cos(theta) -Sin(theta);
        #  Sin(theta)  Cos(theta)]
        # Only rotate by half the angle?
    def rotate(self, offset, angle_type, rescue_sheep):
        
        theta = 0
        
        vector_a = np.array([0,0])
        vector_b = np.array([0,0])

        if(angle_type == "drive"):
            vector_a = np.copy(self.GCM)
            vector_b = np.copy(self.goal)
        
        elif(angle_type == "collect"):
            vector_a = np.copy(rescue_sheep.position)
            vector_b = np.copy(self.GCM)
        
        vector_a = vector_a - self.position
        vector_b = vector_b - self.position

        angle_vector = np.dot(vector_a,vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        theta = np.arccos(angle_vector)

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        offset = np.dot(rotation_matrix,offset)

        return offset