import matplotlib.pyplot as plt

def parasite_or_not(x):
    if (x < 0.5):
        return str('P')
    else:
        return str('U')

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
    
    
def visualize_image(dataset_info, train_dataset, count = 10):
    for i, (image, label) in enumerate(train_dataset.take(count)):
        ax = plt.subplot(4, 4, i + 1)
        
        plt.imshow(image)
        plt.title(dataset_info.features["label"].int2str(label))
        plt.axis("off")
        
    plt.show()
        
def open_model_results(model, test_dataset, count = 9):
    for i, (image, label) in enumerate(test_dataset.take(count)):
        ax = plt.subplot(3, 3, i + 1)
        
        plt.imshow(image[0])
        
        actual_value = str(parasite_or_not(label.numpy()[0]))
        predicted_value = str(parasite_or_not(model.predict(image)[0][0]))
        plt.title(f"{actual_value}:{predicted_value}")
        
        plt.axis("off")
    plt.show()