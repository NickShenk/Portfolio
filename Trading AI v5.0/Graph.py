import matplotlib.pyplot as plt

# Enable interactive mode
plt.ion()
stock = None
vertical_lines = []
def init():
    # Initial setup of the figure and axes
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label=f'{stock} performance this year')  # Line for arr1
    line1.set_color((1.0,0.0,1.0))
    line2, = ax.plot([], [], label='AI performance')  # Line for arr2
    line2.set_color((0.0,.5,1.0))
    ax.set_xlim(0, 365)
    ax.set_ylim(-2, 2)
    ax.legend()
    ax.grid(True)
    # Update function
    global update
    def update(data):
        arr1, arr2,buys,sells = data  # Unpack the arrays
        x = [i for i in range(365)]  # x-values for arr1
        for line in vertical_lines:
            line.remove()
        vertical_lines.clear()
        # Update line data
        line1.set_data(x, arr1)
        line2.set_data(x, arr2)
        ax.set_ylim(min(arr1) - .2, max(arr2) + .2)
        for buy in buys:
            vertical_lines.append(plt.axvline(x=buy, color='green', linestyle='--'))
        for sell in sells:
            vertical_lines.append(plt.axvline(x=sell, color='red', linestyle='--'))
        # Redraw the plot
        fig.canvas.draw()
        fig.canvas.flush_events()