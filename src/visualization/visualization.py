import matplotlib.pyplot as plt

def show_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    
    plt.legend(['train_loss', "val_loss"])
    plt.show()
    
def show_image(image):
    plt.plot()
    plt.imshow(image)
    
    plt.show()