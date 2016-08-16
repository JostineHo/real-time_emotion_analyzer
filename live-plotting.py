import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open('emotion.txt', 'r').read()
    lines = graph_data.split('\n')
    xs = []
    y_angry = []
    y_fear = []
    y_happy = []
    y_sad = []
    y_surprise = []
    y_neutral = []
    for line in lines:
        if len(line) > 1:
            time, angry, fear, happy, sad, surprise, neutral = line.split(',')
            xs.append(time)
            y_angry.append(angry)
            y_fear.append(fear)
            y_happy.append(happy)
            y_sad.append(sad)
            y_surprise.append(surprise)
            y_neutral.append(neutral)

    ax1.clear()
    ax1.plot(xs, y_angry)
    ax1.plot(xs, y_fear)
    ax1.plot(xs, y_happy)
    ax1.plot(xs, y_sad)
    ax1.plot(xs, y_surprise)
    ax1.plot(xs, y_neutral)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
