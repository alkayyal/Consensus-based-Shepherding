
import numpy as np

class Sheep:
    def __init__(self, id, pos, sensing_range, repulsion_range, movement_delta):
        self.id = id # Identification number of sheep
        self.position = pos # Location of Sheep
        self.heading = np.array([0,0]) # Unit vector of movement of sheep
        self.previous_heading = np.array([0,0]) # Previous unit vector of movement
        self.LCM = np.array([0,0]) # Local Center of Mass
        self.connections = [] # Other sheep in sensing range
        self.sensing_range = sensing_range # Radius other sheep and sheepdogs are detected
        self.repulsion_range = repulsion_range # Radius where other sheep repel away
        self.movement_delta = movement_delta # Distance sheep move per timestep
        self.sheepdogs = [] # Sheepdogs within sensing range

    def __repr__(self):
        return str(self.position)
    def __str__(self):
        return str(self.position)

    # Add agents as its neighbors
    def connect(self, agents):
        self.connections = []
        for aa in agents:
            distance_to_sheep = np.linalg.norm(self.position - aa.position)
            if(aa.id != self.id and distance_to_sheep <= self.sensing_range):
                self.connections.append(aa)

    # Returns its neighbor set
    def neighbors(self):
        return self.connections
    
    # Determines and returns set of sheepdogs within the sheeps sensing radius
    def sheepdog_neighbors(self, sheepdog):
        self.sheepdogs = []
        for i in sheepdog:
            distance_to_dogs = np.linalg.norm(self.position - i.position)
            if(distance_to_dogs <= self.sensing_range):
                self.sheepdogs.append(i)
        return self.sheepdogs
    
    # Generates the local center of mass
    def calculate_LCM(self):
        # Triggers flag the first time the function is run to 
        # ensure previous LCM is non zero

        x_avg = self.position[0]
        y_avg = self.position[1]
        for i in self.connections:
            x_avg += i.position[0]
            y_avg += i.position[1]
        
        x_avg /= (len(self.connections) + 1)
        y_avg /= (len(self.connections) + 1)
        
        self.LCM = np.array([x_avg,y_avg])
        return
    
    # Adhering to the heuristics described in 
    # Str¨oombom, D., et al.: Solving the shepherding problem: heuristics for herding
    # autonomous, interacting agents. J. R. Soc. Interface 11(100), 20140719 (2014)
    def update_position(self):
        self.previous_heading = np.copy(self.heading)
        

        #Noise term
        noise = np.array([0,0])
        noise = np.array([np.random.normal(),np.random.normal()])
        noise /= np.linalg.norm(noise)

        #Tuning parameters
        #values gotten from paper mentioned above
        h = 0.5
        pa = 2
        c = 1.05
        ps = 1
        epsilon = 0.1
        
        # Repulsion of sheep with sheepdogs in range
        # RiSj is the repulsion vector between the ith sheep and the jth shepherd
        # RiSj = 1/dist(Ai - Sj) * (Ai - Sj)
        # sum all of the terms to get RiS
        # Normalize the term
        # Sheep will stay in position if no sheepdog is in range
        RiS = np.array([0,0])
        if(len(self.sheepdogs) > 0):
            for i in self.sheepdogs:
                vector_difference = self.position - i.position
                magnitude = np.linalg.norm(vector_difference)
                RiSj = 1/magnitude * vector_difference
                RiS = RiS + RiSj
            RiS = RiS / np.linalg.norm(RiS)
        
        #Attraction to Local Center of Mass
        #Calculate latest LCM
        #Ci is the attraction vector between the LCM and the ith Sheep
        #Ci = LCM - Ai
        #Normalize the term
        self.calculate_LCM()
        Ci = np.array([0,0])
        if(self.LCM[0] != self.position[0] and self.LCM[1] != self.position[1]):
            Ci = self.LCM - self.position
            # print(f"id:{self.id}, Ci:{Ci}")
            Ci = Ci / np.linalg.norm(Ci)

        #Repulsion of sheep with other sheep
        #Riaj is the repulsion vector between the ith sheep and the jth sheep
        #Riaj = 1/dist(Ai - Aj) * (Ai - Aj)
        #Sum all of the terms to get Ria
        #Normalize the term
        Ria = np.array([0.0, 0.0])
        if(len(self.connections) > 0):
            # print(self.connections)
            for i in self.connections:
                vector_difference = self.position - i.position
                distance_to_sheep = np.linalg.norm(self.position - i.position)
                # print(f"id:{i.id}, dist:{distance_to_sheep}")
                if(distance_to_sheep <= self.repulsion_range):
                    Riaj = 1/distance_to_sheep * vector_difference
                    Ria += Riaj
            if(Ria[0] == 0 and Ria[1] == 0):
                pass
            else:
                Ria = Ria / np.linalg.norm(Ria)

        #Tuning each term
        momentum_term = h * self.previous_heading
        center_of_mass_attraction_term = c * Ci
        sheep_repulsion_term = pa * Ria
        sheepdog_repulsion_term = ps * RiS
        noise_term = epsilon * noise
        
        #Heading is a linear combination of the three vectors calculated
        self.heading = momentum_term + center_of_mass_attraction_term + sheep_repulsion_term + sheepdog_repulsion_term + noise_term
        # if(self.heading[0])
        self.heading /= np.linalg.norm(self.heading)

        #If a sheep cannot see the shepherd then it grazes
        if(len(self.sheepdogs) <= 0):
            self.heading = 1.5 * h * self.previous_heading + noise_term

        self.position += self.movement_delta * self.heading

        # print("Momentum:",momentum_term)
        # print("COM:",center_of_mass_attraction_term)
        # print("Sheep Repulsion:",sheep_repulsion_term)
        # print("Sheepdog Repulsion:",sheepdog_repulsion_term)
        # print("Noise:",noise_term)
        return















