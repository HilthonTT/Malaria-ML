import matplotlib.pyplot as plt

def show_loss(history):
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])

    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    # Set y-axis limits to improve visualization
    plt.ylim([0, max(max(history.history['loss']), max(history.history['val_loss']))])
    
    plt.legend(['Training Loss', "Validation Loss"])
    plt.show()
    
def show_image(image):
    plt.plot()
    plt.imshow(image)
    
    plt.show()