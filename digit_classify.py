import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import pygame as pg
from tkinter import messagebox


# load digit_classifier model
model_filename = 'digit_classifier_modified.sav'
digit_classifier = pickle.load(open(model_filename, 'rb'))

messagebox.showinfo("Instruction", "Draw some digit and I'll guess that number \n 1. Hold left click and drag to draw a number \n 2. Press Space to see what I've guessed")

def processImage(path):
    
    image = Image.open(path)
    image = image.resize((28,28))
    img_array = np.asarray(image)
   
    # convert image to grayscale
    img_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    
    # invert color of image
    img_array = 255 - img_array
    
    # scale the image
    img_array = img_array / 255
    
    return img_array

def init():
    

    pg.init()
    
    screen = pg.display.set_mode([800, 600])
    clock = pg.time.Clock()
    is_running = True
    mouse_down = False
    
    while is_running:
        
        x, y = pg.mouse.get_pos()

        for event in pg.event.get():

            # if quit event is raised exit the window
            if event.type == pg.QUIT:
                is_running = False
            
            elif event.type == pg.MOUSEBUTTONDOWN:
                mouse_down = True
            
            elif event.type == pg.MOUSEBUTTONUP:
                mouse_down = False
            
        
        color = (255,255,255)
        
        # Drawing Rectangle
        if mouse_down:
            pg.draw.rect(screen, color, pg.Rect(x,y,25,25))
        
        # predict when space bar is pressed
        if pg.key.get_pressed()[pg.K_SPACE]:

            pg.image.save(screen, "digit.jpeg")

            image = processImage('digit.jpeg')

            plt.imshow(image)
            y_predict = digit_classifier.predict(np.array([image]))
            print(np.argmax(y_predict))

            messagebox.showinfo("Predicted digit: ",str(np.argmax(y_predict)))

            screen.fill('black')

        
        pg.display.flip()

        # set fps
        clock.tick(60.0)

init()