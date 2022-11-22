import pgzrun

TITLE = 'Flappy Bird'
WIDTH = 400
HEIGHT = 708
GRAVITY = 0.3

def update():
    barry_the_bird.y += barry_the_bird.speed
    barry_the_bird.speed += GRAVITY
    top_pipe.x += scroll_speed
    bottom_pipe.x += scroll_speed
    if top_pipe.right < 0:
        top_pipe.left = WIDTH
        bottom_pipe.left = WIDTH
    if barry_the_bird.y > HEIGHT or barry_the_bird.y < 0:
        reset()

def draw():
    screen.blit('background', (0, 0))
    barry_the_bird.draw()
    top_pipe.draw()
    bottom_pipe.draw()

def on_mouse_down():
    print ('The mouse was clicked')
    barry_the_bird.speed = -6.5
def reset():
    print ("Back to the start...")
    barry_the_bird.speed = 1
    barry_the_bird.center = (75, 350)
    top_pipe.center = (300, 0)
    bottom_pipe.center = (300, top_pipe.height + gap)

barry_the_bird = Actor('bird1', (75, 350))
barry_the_bird.speed = 1

gap = 140
top_pipe = Actor('top', (300, 0))
bottom_pipe = Actor('bottom', (300, top_pipe.height + gap))

scroll_speed = -1

pgzrun.go()