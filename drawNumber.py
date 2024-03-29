import pygame
import tensorflow as tf
import numpy as np
from tkinter import *
from tkinter import messagebox

class pixel(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255,255,255)
        self.neighbors = []

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.x + self.width, self.y + self.height))

    def getNeighbors(self, g):
        j = self.x // 20
        i = self.y // 20
        rows = 28
        cols = 28

        # Horizontal and vertical neighbors
        if i < cols - 1:  # Right
            self.neighbors.append(g.pixels[i + 1][j])
        if i > 0:  # Left
            self.neighbors.append(g.pixels[i - 1][j])
        if j < rows - 1:  # Up
            self.neighbors.append(g.pixels[i][j + 1])
        if j > 0:  # Down
            self.neighbors.append(g.pixels[i][j - 1])

        if j > 0 and i > 0:
            self.neighbors.append(g.pixels[i - 1][j - 1])

        if j + 1 < rows and i > -1 and i - 1 > 0:
            self.neighbors.append(g.pixels[i - 1][j + 1])

        if j - 1 < rows and i < cols - 1 and j - 1 > 0:
            self.neighbors.append(g.pixels[i + 1][j - 1])

        if j < rows - 1 and i < cols - 1:
            self.neighbors.append(g.pixels[i + 1][j + 1])


class grid(object):
    pixels = []

    def __init__(self, row, col, width, height):
        self.rows = row
        self.cols = col
        self.len = row * col
        self.width = width
        self.height = height
        self.generatePixels()
        pass

    def draw(self, surface):
        for row in self.pixels:
            for col in row:
                col.draw(surface)

    def generatePixels(self):
        x_gap = self.width // self.cols
        y_gap = self.height // self.rows
        self.pixels = []
        for r in range(self.rows):
            self.pixels.append([])
            for c in range(self.cols):
                self.pixels[r].append(pixel(x_gap * c, y_gap * r, x_gap, y_gap))

        for r in range(self.rows):
            for c in range(self.cols):
                self.pixels[r][c].getNeighbors(self)

    def clicked(self, pos):
        try:
            t = pos[0]
            w = pos[1]
            g1 = int(t) // self.pixels[0][0].width
            g2 = int(w) // self.pixels[0][0].height

            return self.pixels[g2][g1]
        except:
            pass

    def convert_binary(self):
        drawing = self.pixels

        Newmatrix = [[] for x in range(len(drawing))]

        for i in range(len(drawing)):
            for j in range(len(drawing[i])):
                if drawing[i][j].color == (255,255,255):
                    Newmatrix[i].append(0)
                else:
                    Newmatrix[i].append(1)

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = tf.keras.utils.normalize(x_test, axis=1)
        for row in range(28):
            for x in range(28):
                x_test[0][row][x] = Newmatrix[row][x]

        return x_test[:1]


def guess(drawing):
    model = tf.keras.models.load_model('m.model')

    predictions = model.predict(drawing)
    print(predictions[0])
    number = (np.argmax(predictions[0]))
    print("Tahminim Numara Şu:", number)
    window = Tk()
    window.withdraw()
    messagebox.showinfo("Tahmin", "Tahminim Numara Şu: " + str(number))
    window.destroy()
    #plt.imshow(li[0], cmap=plt.cm.binary)
    #plt.show()


def main():
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                drawing = g.convert_binary()
                guess(drawing)
                g.generatePixels()
            if pygame.mouse.get_pressed()[0]:

                pos = pygame.mouse.get_pos()
                clicked = g.clicked(pos)
                clicked.color = (0,0,0)
                for n in clicked.neighbors:
                    n.color = (0,0,0)

            if pygame.mouse.get_pressed()[2]:
                try:
                    pos = pygame.mouse.get_pos()
                    clicked = g.clicked(pos)
                    clicked.color = (255,255,255)
                except:
                    pass

        g.draw(win)
        pygame.display.update()

pygame.init()
width = height = 560
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Numara Tahminleyici/Cizimden Sonra Space tuslayın")
g = grid(28, 28, width, height)
main()


pygame.quit()
quit()
