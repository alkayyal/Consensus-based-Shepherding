
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Sheep import *
from Sheepdog import *

def create_sheep(sheep_size,corner_coordinate,sensing,repel,movement_speed):
    sheep = []
    # for i in range(sheep_size):
    #     starting_x = np.random.rand()*corner_coordinate[0]//2 + corner_coordinate[0]//2
    #     starting_y = np.random.rand()*corner_coordinate[1]//2 
    #     # print(f"x:{starting_x}, y:{starting_y}")
    #     # starting_x = 0.0
    #     # starting_y = 30.0
    #     start = np.array([starting_x,starting_y])

    #     molly = Sheep(i,start,sensing//2,repel,movement_speed)
    #     sheep.append(molly)

    sheep.append(Sheep(0,np.array([10.0,20.0]),sensing//2,repel,movement_speed))
    sheep.append(Sheep(1,np.array([20.0,20.0]),sensing//2,repel,movement_speed))
    sheep.append(Sheep(2,np.array([20.0,10.0]),sensing//2,repel,movement_speed))

    return sheep

def true_GCM(sheep):
    coord = np.array([0,0])
    for i in sheep:
        coord = coord + i.position
    coord = coord / len(sheep)
    # print(f"true gcm: {coord}")
    return coord

def simulation(sheepdogs_size, sheep_size, time, goal, starting_box, sensing, movement_speed, repel):

    # Initialize sheepdog agents to random coordinates in quadrant 1 (lower left quadrant) of starting box
    sheepdogs = [Sheepdog(0, sensing, np.array([10.0,10.0]), goal, sheep_size, movement_speed * 1.5)]
    
    # Initialize sheep agents to random coordinates in quadrant 3 (upper right quadrant) of starting box
    sheep = create_sheep(sheep_size, starting_box[1], sensing, repel, movement_speed)

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
            # print(sheep)
            s.sheepdog_neighbors(sheepdogs)
            s.update_position()
            sheep_pos.append(np.copy(s.position))

        temp.append(np.copy(true_GCM(sheep)))

        # dog_gcm.append([temp[0],temp[1],temp[2],temp[3]])
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
        scatter_green = ax.scatter(goal_x, goal_y, color='green', label='Goal', s=50)

        # scatter_green = ax.scatter(1000, 1000, color='green', label='Goal', s=50)

        # Add both scatter plots to the frame
        frames.append([scatter_blue, scatter_red, scatter_green])

    # Create the animation
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=True)

    # # Show the animation
    plt.show()
    # plt.title("Homogenous Consensus")
    # ani.save("Heterogenous_Consensus_movement_and_sensing.gif", writer="pillow", fps=10)
    return


def main():
    # Number of sheepdog agents
    D = 1

    # Number of Sheep agents
    N = 1

    # Goal coordinates
    goal = np.array([40,40])

    # Starting box
    # Upper Left corner
    upper_left = np.array([0,0])

    # Lower Right corner
    lower_right = np.array([51,51])

    # Time steps. Feel free to change
    time = 300

    # Sensing Range
    sensing = 50

    # Distance agent can travel in a single timestep
    # Doubled for Sheep
    # Tripled for Sheepdog
    movement_speed = 1

    # Range Sheep will move away from each other
    repel = 2

    # Run the simulation and store the movement history
    frames,dog_gcm = simulation(D,N,time, goal, (upper_left,lower_right),sensing, movement_speed, repel)
    plot_the_data(goal,frames,lower_right)

    return


if __name__ == "__main__":
    main()









