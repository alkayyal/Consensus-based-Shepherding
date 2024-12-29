import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Sheep import *
from Sheepdog import *

# def kcirculant(n,k):
#     edges = []
#     for i in range(n):
#         for j in range(1, k + 1):
#             edges.append((i, (i + j) % n))
#     return edges

def create_sheepdog(sheepdogs_size,sheep_size,goal,corner_coordinate,sensing,movement_speed):
    sheepdogs = []
    for i in range(sheepdogs_size):
        starting_x = np.random.rand()*corner_coordinate[0]//2
        starting_y = np.random.rand()*corner_coordinate[1]//2 + corner_coordinate[1]//2
        # print(f"x:{starting_x}, y:{starting_y}")
        start = np.array([starting_x,starting_y])
        lassy = Sheepdog(i,sensing*(sheepdogs_size-i)*0.5,start,goal,sheep_size,movement_speed* (i+1) * 0.5)#
        sheepdogs.append(lassy)
    
    return sheepdogs

def create_sheep(sheep_size,corner_coordinate,sensing,repel,movement_speed):
    sheep = []
    for i in range(sheep_size):
        starting_x = np.random.rand()*corner_coordinate[0]//2 + corner_coordinate[0]//2
        starting_y = np.random.rand()*corner_coordinate[1]//2 
        # print(f"x:{starting_x}, y:{starting_y}")
        start = np.array([starting_x,starting_y])

        molly = Sheep(i,start,sensing//2,repel,movement_speed)
        sheep.append(molly)
    return sheep

def true_GCM(sheep):
    coord = np.array([0,0])
    for i in sheep:
        coord = coord + i.position
    coord = coord / len(sheep)
    # print(f"true gcm: {coord}")
    return coord

def simulation(sheepdogs_size, sheep_size, time, goal, starting_box):

    # Make the them homogenous for now
    sensing = 50
    movement_speed = 1
    repel = 4

    # Initialize sheepdog agents to random coordinates in quadrant 1 (lower left quadrant) of starting box
    sheepdogs = create_sheepdog(sheepdogs_size,sheep_size,goal,starting_box[1],sensing,movement_speed*1.5)
    

    # Initialize sheep agents to random coordinates in quadrant 3 (upper right quadrant) of starting box
    sheep = create_sheep(sheep_size,starting_box[1],sensing,repel,movement_speed)

    # Storing the sheep and sheepdog positions at every time step
    frames = []

    dog_gcm = []

    for i in range(time):
        # Positions of Sheep and Sheepdogs at current time step
        sheep_pos = []
        dog_pos = []
        
        temp = []

        # Reestablish sheepdog network
        # Reestablish sheep network
        # Update position and record coordinates
        for d in sheepdogs:
            d.connect(sheepdogs)
            d.sheep_neighbors(sheep)
            d.update_position()
            dog_pos.append(np.copy(d.position))
            temp.append(np.copy(d.get_GCM()))
            # print(f"dogs GCM: {d.get_GCM()}")
            # print("hello")

        # Restablish sheep network
        # Restablish sheepdog network
        # Update position and record coordinates
        for s in sheep:
            s.connect(sheep)
            s.sheepdog_neighbors(sheepdogs)
            s.update_position()
            sheep_pos.append(np.copy(s.position))

        temp.append(np.copy(true_GCM(sheep)))

        dog_gcm.append([temp[0],temp[1],temp[2],temp[3]])
        # print(([temp[0],temp[1],temp[2],temp[3]]))
        
        frames.append([sheep_pos,dog_pos])

    return frames, dog_gcm

def plot_the_data(goal, frame, border):
    # Generate a figure and axes
    fig, ax = plt.subplots()
    bounds_x = border[0]
    bounds_y = border[1]
    ax.set_xlim(-bounds_x//2, bounds_x * 1.5)
    ax.set_ylim(-bounds_y//2, bounds_y * 1.5)

    frames = []

    for i in frame:
        
        sheep_x = [point[0] for point in i[0]]
        sheep_y = [point[1] for point in i[0]]

        dog_x = [point[0] for point in i[1]]
        dog_y = [point[1] for point in i[1]]
        
        goal_x = goal[0]
        goal_y = goal[1]
            
        scatter_blue = ax.scatter(sheep_x, sheep_y, color='blue', label='Sheep', s=50)
        scatter_red = ax.scatter(dog_x, dog_y, color='red', label='Sheepdogs', s=50)
        scatter_green = ax.scatter(55, 32, color='green', label='Goal', s=50)

        # Add both scatter plots to the frame
        frames.append([scatter_blue, scatter_red, scatter_green])

    # Create the animation
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=True)

    # # Show the animation
    plt.show()
    # plt.title("Homogenous Consensus")
    # ani.save("Heterogenous_Consensus_movement_and_sensing.gif", writer="pillow", fps=10)
    return

def stats_plot(dog_gcm):
    true_val = []
    dog_1 = []
    dog_2 = []
    dog_3 = []
    for i in dog_gcm:
        true_val.append(i[3])
        dog_1.append(i[0])
        dog_2.append(i[1])
        dog_3.append(i[2])

    t = np.arange(0, 100, 1)

    true_val_x = [point[0] for point in true_val]
    true_val_y = [point[1] for point in true_val]
    
    dog1_x = [point[0] for point in dog_1]
    dog1_y = [point[1] for point in dog_1]

    dog2_x = [point[0] for point in dog_2]
    dog2_y = [point[1] for point in dog_2]

    dog3_x = [point[0] for point in dog_3]
    dog3_y = [point[1] for point in dog_3]

    plt.plot(true_val_x,true_val_y, c = 'g', label = "True GCM")
    plt.plot(dog1_x,dog1_y, c = 'r', label = "Dog 1")
    plt.plot(dog2_x,dog2_y, c = 'b', label = "Dog 2")
    plt.plot(dog3_x,dog3_y, c = 'y', label = "Dog 3")

    plt.title("Heterogeneous Consensus - Varied Movement and Sensing")
    plt.legend()

    plt.show()


    plt.plot(t,true_val_x, c = 'g', label = "True GCM")
    plt.plot(t,dog1_x, c = 'r', label = "Dog 1")
    plt.plot(t,dog2_x, c = 'b', label = "Dog 2")
    plt.plot(t,dog3_x, c = 'y', label = "Dog 3")

    plt.title("Heterogeneous Consensus - Varied Movement and Sensing")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("X Position")
    plt.show()

    #Plotting y axis with true gcm 
    plt.plot(t,true_val_y, c = 'g', label = "True GCM")
    plt.plot(t,dog1_y, c = 'r', label = "Dog 1")
    plt.plot(t,dog2_y, c = 'b', label = "Dog 2")
    plt.plot(t,dog3_y, c = 'y', label = "Dog 3")

    plt.title("Heterogeneous Consensus - Varied Movement and Sensing")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("Y Position")

    plt.show()



def main():
    # Number of sheepdog agents
    D = 3

    # Number of Sheep agents
    N = 60

    # Goal coordinates
    goal = np.array([0,0])

    # Starting box
    # Upper Left corner
    upper_left = np.array([0,0])

    # Lower Right corner
    lower_right = np.array([51,51])

    # Time steps. Feel free to change
    time = 100

    # Run the simulation and store the movement history
    frames,dog_gcm = simulation(D,N,time, goal, (upper_left,lower_right))
    plot_the_data(goal,frames,lower_right)

    # stats_plot(dog_gcm)
    return


if __name__ == "__main__":
    main()