# Import numpy to work with arrays
import numpy as np

# Import os to work with image directories
import os

# For visualization
import matplotlib.pyplot as plt

# For randomly choosing samples from datasets
import random

# For creating the gui window
import tkinter as tk

# For upgrading the gui design to a modern look
import customtkinter as ctk

# Import PIL to work with images
from PIL import Image, ImageOps

# Import K neihbours method from scikit learn
from sklearn.neighbors import KNeighborsRegressor

# Import KMeans method from scikit learn
from sklearn.cluster import KMeans

# Import LinearSVC method from scikit learn
from sklearn.svm import LinearSVC


# Define a function which will simply clean the previous page's widgets
def destroy_previous_widgets():
    # For every active widget of the window
    for i in window.winfo_children():
        # Destroy the widget
        i.destroy()

# Define the background color of the interface
background_color = ("#ded9e0", "#121212")

# Define font for the ABCs
ABCs_font = ("Roboto", 50)
    
# Create the GUI window
window = ctk.CTk()

# Name the GUI window
window.title("ABCs of Machine Learning")

# Define the width and height of the interface
width, height = 1920, 1080

# Set the interface window size
window.geometry(f"{width}x{height}")

# Define the path to datasets and to-be-created-while-using-the-interface images
input_dir_today = "datasets/today_faces"
input_dir_renaissance = "datasets/renaissance_faces"

# Save the list of files from datasets to lists
today_dataset = os.listdir(input_dir_today)
renaissance_dataset = os.listdir(input_dir_renaissance)

# Define function which displays images from datasets in a format 5 images/columns per row
def show_dataset_images(dataset, input_dir, frame):
    # Define the initial cell
    row = 0
    column = 0
    # For every image in a dataset to visualize
    for file in dataset:
        # Specify the file path
        img_path = os.path.join(input_dir, file)
        # Open the image in the gui
        image = ctk.CTkImage(light_image=Image.open(img_path), size=(64, 64))
        # Create a widget of that image
        image_label = ctk.CTkLabel(frame, image=image, text="")
        # Place that image to the next cell
        image_label.grid(row=row, column=column)

        # If there are columns still left in the row
        if column < 4:
            # Make the next image be in the next free column of that row
            column += 1
        # If there are no more free columns int the row
        elif column == 4:
            # Move to the next row
            row += 1
            column = 0

# Create a function which will generate a main frame for a page which will host subframes
def create_main_frame():
    # Create the frame
    main_frame = ctk.CTkFrame(window, fg_color=background_color)
    # Place it to the GUI window
    main_frame.pack(fill="both", expand=1) # fill "both" means horizontally and vertically

    # Define the grid layout for the main frame
    for i in range(3):
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(i, weight=1)

    # Return the frame as a variable so it can be used by respective pages
    return main_frame

# Define a function which displays joined dataset based on the percentages chosen on the picture completion page
def display_joined_dataset():
    # Make the main frame available to this function
    global main_frame

    # Define a list which will store the paths to all the images in the joined dataset
    file_paths = []
    # For every file chosen from the renaissance dataset
    for file in renaissance_input_samples:
        # Specify the file path
        file_path = os.path.join(input_dir_renaissance, file)
        # Add the file path to the list
        file_paths.append(file_path)
    # For every file chosen from the today dataset
    for file in today_input_samples:
        # Specify the file path
        file_path = os.path.join(input_dir_today, file)
        # Add the file path to the list
        file_paths.append(file_path)
    # Shuffle the file paths so that the images from two original datasets get mixed
    random.shuffle(file_paths)

    # Define the initial cell
    row = 0
    column = 0
    # For every image in a dataset to visualize
    for i in range(len(file_paths)):
        # Specify the file path
        img_path = file_paths[i]
        # Open the image in the gui
        image = ctk.CTkImage(light_image=Image.open(img_path), size=(64, 64))
        # Create a widget of that image
        image_label = ctk.CTkLabel(number_of_clusters_frame, image=image, text="")
        # Place that image to the next cell
        image_label.grid(row=row, column=column)
        # If there are columns still left in the row
        if column < 4:
            # Make the next image be in the next free column of that row
            column += 1
        # If there are no more free columns int the row
        elif column == 4:
            # Move to the next row
            row += 1
            column = 0

# Define a function which displays either all or currently used images from todays dataset
def display_todays_dataset(dataset, input_dir):
    # Make the main frame available to this function, make the todays dataset frame available to other functions
    global main_frame, todays_dataset_frame

    # Define a frame for the today dataset section
    todays_dataset_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
    # Place the today dataset sectionn to the window
    todays_dataset_frame.grid(row=0, column=0)

    # Define the grid layout for todays dataset
    for i in range(10):
        todays_dataset_frame.rowconfigure(i, weight=1)
    for i in range(5):
        todays_dataset_frame.columnconfigure(i, weight=1)

    # Display the images of the currently used pictures from today's dataset
    show_dataset_images(dataset, input_dir, todays_dataset_frame)

# Define a function which display either all or currently used images from the renaissance dataset
def display_renaissance_dataset(dataset, input_dir):
    # Make the main frame available to this function, make the renaissance dataset frame available to other functions
    global main_frame, renaissance_dataset_frame
    # Define a frame for the renaissance dataset section
    renaissance_dataset_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
    # Place the renaissance dataset section to the window
    renaissance_dataset_frame.grid(row=0, column=2)

    # Define the grid layout for renaissance dataset section
    for i in range(10):
        renaissance_dataset_frame.rowconfigure(i, weight=1)
    for i in range(5):
        renaissance_dataset_frame.columnconfigure(0, weight=1)

    # Display the images of the currently used pictures from the renaissance dataset
    show_dataset_images(dataset, input_dir, renaissance_dataset_frame)

# Define the C guide page
def goodbye_page():
    # Make the main frame available to use for the goodbye page
    global main_frame

    # Destroy previous widgets
    destroy_previous_widgets()

    # Create main frame for subframes to use
    main_frame = create_main_frame()

    # Define a frame for goodbye page
    goodbye_page_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
    # Place the A guide page frame
    goodbye_page_frame.grid(row=0, column=1)

    # Define the grid layout for goodbye page
    for i in range(10):
        goodbye_page_frame.rowconfigure(i, weight=1)
    goodbye_page_frame.columnconfigure(0, weight=1)

    # Create goodbye label
    goodbye_label = ctk.CTkLabel(goodbye_page_frame,
                                 text="That's it! Now you know the ABCs of machine learning. Stay curious and explore more!"
                                      "\nIf you wish to repeat this process, click on 'Restart'.",
                                 wraplength=500)
    # Place the goodbye label to the grid
    goodbye_label.grid(row=9, column=1)

    # Define a button which goes to the beginning
    restart_button = ctk.CTkButton(goodbye_page_frame,
                                       text="Restart",
                                       command=welcome_page)
    # Place the button to the grid
    restart_button.grid(row=10, column=1)

# Define the image classification page
def classification_page():
    # Destroy previous widgets
    destroy_previous_widgets()

    # Create main frame for subframes to use
    main_frame = create_main_frame()

    # Define a frame for the classification page
    classification_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
    # Place the frame for the classification page
    classification_frame.grid(row=0, column=1)

    # Define the grid layout for the view of clusters
    for i in range(9):
        classification_frame.rowconfigure(i, weight=1)
    for i in range(3):
        classification_frame.columnconfigure(i, weight=1)

    # Create dictionary to store which cluster/class has which user-generated name
    class_pairs = {}

    # Define a function which will take 3 random images from each original dataset and generate test data
    # for imagine clasification classifier
    def generate_test_data():

        # Define a list into which test data will be put
        test_data = []

        # Define a list into which test data file paths will be put for later display of those images in results
        test_data_file_paths = []

        # Choose 3 files from todays dataset randomly
        todays_dataset_test_images = random.sample(today_dataset, 3)
        # Choose 3 files from renaissance dataset randomly
        renaissance_dataset_test_images = random.sample(renaissance_dataset, 3)

        # For every 3 files randomly taken from todays dataset
        for file in todays_dataset_test_images:
            filepath = os.path.join(input_dir_today, file)
            # Load the image
            img = Image.open(filepath)
            # Turn the image to grayscale
            img = ImageOps.grayscale(img)
            # Turn the image into array (2D)
            img = np.asarray(img)
            # Make the 2D array 1D array
            img = img.flatten()
            # Append the array to the test data list
            test_data.append(img)
            # Append the filepaths to the test data path list
            test_data_file_paths.append(filepath)

        # For every 3 files randomly taken from renaissance dataset
        for file in renaissance_dataset_test_images:
            filepath = os.path.join(input_dir_renaissance, file)
            # Load the image
            img = Image.open(filepath)
            # Turn the image to grayscale
            img = ImageOps.grayscale(img)
            # Turn the image into array (2D)
            img = np.asarray(img)
            # Make the 2D array 1D array
            img = img.flatten()
            # Append the array to the test data list
            test_data.append(img)
            # Append the filepaths to the test data path list
            test_data_file_paths.append(filepath)

        # Transform the test data into 1D array
        test_data = np.asarray(test_data)

        # return the testing data and its file paths
        return test_data, test_data_file_paths

    def save_classes():
        # Define a list which will store unfilled class names
        unfilled = []

        # If there is cluster 0
        if "class0_value_entry" in globals():
            # Check if name for it was inputted
            if len(class0_value_entry.get()) == 0:
                # If not, store this cluster as having incomplete
                unfilled.append(0)
            # If the name was inputted
            else:
                # Store this name as name for cluster 0
                class_pairs[0] = class0_value_entry.get()

        # Comments for cluster 0 apply, find cluster 0 procedure above
        if "class1_value_entry" in globals():
            if len(class1_value_entry.get()) == 0:
                unfilled.append(1)
            else:
                class_pairs[1] = class1_value_entry.get()

        # Comments for cluster 0 apply, find cluster 0 procedure above
        if "class2_value_entry" in globals():
            if len(class2_value_entry.get()) == 0:
                unfilled.append(2)
            else:
                class_pairs[2] = class2_value_entry.get()

        # Comments for cluster 0 apply, find cluster 0 procedure above
        if "class3_value_entry" in globals():
            if len(class3_value_entry.get()) == 0:
                unfilled.append(3)
            else:
                class_pairs[3] = class3_value_entry.get()

        # Comments for cluster 0 apply, find cluster 0 procedure above
        if "class4_value_entry" in globals():
            if len(class4_value_entry.get()) == 0:
                unfilled.append(4)
            else:
                class_pairs[4] = class4_value_entry.get()

        # Comments for cluster 0 apply, find cluster 0 procedure above
        if "class5_value_entry" in globals():
            if len(class5_value_entry.get()) == 0:
                unfilled.append(5)
            else:
                class_pairs[5] = class5_value_entry.get()

        # Comments for cluster 0 apply, find cluster 0 procedure above
        if "class6_value_entry" in globals():
            if len(class6_value_entry.get()) == 0:
                unfilled.append(6)
            else:
                class_pairs[6] = class6_value_entry.get()

        # Comments for cluster 0 apply, find cluster 0 procedure above
        if "class7_value_entry" in globals():
            if len(class7_value_entry.get()) == 0:
                unfilled.append(7)
            else:
                class_pairs[7] = class7_value_entry.get()

        # Comments for cluster 0 apply, find cluster 0 procedure above
        if "class8_value_entry" in globals():
            if len(class8_value_entry.get()) == 0:
                unfilled.append(8)
            else:
                class_pairs[8] = class8_value_entry.get()

        # If there are some unfilled classes
        if len(unfilled) != 0:
            # Remind the user to fill them in
            # Add 1 to each item in the incomplete list so that eg. cluster 0 is display as cluster 1
            unfilled = [i + 1 for i in unfilled]
            # Join the numbers of unfilled classes into one string
            incomplete_classes = ",".join(str(x) for x in unfilled)
            # Show the unfilled classes in a label
            error_label.configure(text=f"Clusters {incomplete_classes} have no names. Please add them.")
        # If all classes have been fulfilled
        else:
            # Create a list to store the images
            data = []
            # Create a list to store the clusters to which respective images belong
            labels = []

            # Transform the original array with image cluster information to list
            clusters_of_images_list = Z.tolist()

            # TRAIN DATA
            # For every file in all the clusters
            for idx, file in enumerate(data_file_paths):
                # Load the image
                img = Image.open(file)
                # Turn the image to grayscale
                img = ImageOps.grayscale(img)
                # Turn the image into array (2D)
                img = np.asarray(img)
                # Make the 2D array 1D array
                img = img.flatten()
                # Append the array to the data list
                data.append(img)
                # Append the cluster of the respective image to the labels list
                labels.append(clusters_of_images_list[idx])


            # Turn the data into an array
            data = np.asarray(data)
            # Turn the labels into an array
            labels = np.asarray(labels)

            # Create a classifier object
            classifier = LinearSVC(random_state = 0)
            # Train the classifier
            classifier.fit(data, labels)

            # TEST DATA
            # generate test data and get file paths of those images too
            test_data, test_data_file_paths = generate_test_data()

            # Generate predictions for the images based on the trained classifier
            predictions = classifier.predict(test_data)

            # Delete the previous section to display results
            classification_frame.destroy()

            # Define a frame for the classification page
            classification_results_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
            # Place the frame for the classification page
            classification_results_frame.grid(row=0, column=1)

            # Define the grid layout for the results of classification
            for i in range(9):
                classification_results_frame.rowconfigure(i, weight=1)
            for i in range(3):
                classification_results_frame.columnconfigure(i, weight=1)

            # Create the column description
            column_predicted_class_label = ctk.CTkLabel(classification_results_frame, text="Predicted cluster")
            # Place the column descriptions
            column_predicted_class_label.grid(row=0, column=2)

            # For 6 images which we will show the result on
            for i in range(6):
                img_path = test_data_file_paths[i]
                # Open the image in the gui
                image_class_page = ctk.CTkImage(light_image=Image.open(img_path), size=(96, 96))
                # Create a widget of that image
                image_class_page_label = ctk.CTkLabel(classification_results_frame, image=image_class_page, text="")
                # Place that image
                image_class_page_label.grid(row=1+i, column=0)

                # Get the predicted cluster code of that image
                predicted_class_code = predictions[i]
                # Get the name the user gave to this cluster
                predicted_class_name = class_pairs[predicted_class_code]
                # Show that cluster user-inputted name through label widget
                predicted_class_label = ctk.CTkLabel(classification_results_frame, text=predicted_class_name)
                # Place the widget
                predicted_class_label.grid(row=1 + i, column=2)

            # Define a button which saves the values
            predict_again_button = ctk.CTkButton(classification_results_frame,
                                                text="Predict for new images",
                                                command=save_classes)
            # Place the button in the clusters section
            predict_again_button.grid(row=9, column=1)

            # Define a button which goes too goodbye screen
            go_to_goodbye_page_button = ctk.CTkButton(classification_results_frame,
                                                 text="Next",
                                                 command=goodbye_page)
            # Place the button in the clusters section
            go_to_goodbye_page_button.grid(row=10, column=1)

    # Display the clusters
    # Define the initial cell
    row = 0
    column = 0

    # For every cluster
    for cluster_no, cluster in enumerate(filenames_of_clusters):
        # Specify the file path
        img_path = cluster
        # Open the image in the gui
        image = ctk.CTkImage(light_image=Image.open(img_path), size=(300, 300))
        # Create a widget of that image
        image_label = ctk.CTkLabel(classification_frame, image=image, text="")
        # Place that image to the next cell
        image_label.grid(row=row, column=column)
        # Define a variable which will store the class value (variable name has to be dynamic to be different
        # in each for loop iteration)
        globals()[f"class{cluster_no}_value_entry"] = tk.StringVar()
        # Define entry box for class
        class_value_entry_box = ctk.CTkEntry(classification_frame,
                                                textvariable=globals()[f"class{cluster_no}_value_entry"],
                                                width=200)
        # Place entry box for class to the grid
        class_value_entry_box.grid(row=row+1, column=column)
        # If there are columns still left in the row
        if column < 4:
            # Make the next image be in the next free column of that row
            column += 1
        # If there are no more free columns int the row
        elif column == 4:
            # Move to 3 rows down
            row += 3
            column = 0

    # Define a button which saves the values
    save_classes_button = ctk.CTkButton(classification_frame,
                                        text="Save values and predict",
                                        command=save_classes)
    # Place the button in the clusters section
    save_classes_button.grid(row=8, column=0, columnspan=3)

    # Define a label which shows if values for classes are not complete when saving
    error_label = ctk.CTkLabel(classification_frame,
                                               text="")
    # Place the slider to the number of clusters frame
    error_label.grid(row=9, column=2)

# Define the C guide page
def C_guide():
    # Make the main frame available to use for the C guide page
    global main_frame

    # Destroy previous widgets
    destroy_previous_widgets()

    # Create main frame for subframes to use
    main_frame = create_main_frame()

    # Define a frame for C guide page
    C_guide_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
    # Place the A guide page frame
    C_guide_frame.grid(row=0, column=1)

    # Define the grid layout for A guide page
    for i in range(10):
        C_guide_frame.rowconfigure(i, weight=1)
    C_guide_frame.columnconfigure(0, weight=1)

    # Create the A text widget
    C_label = ctk.CTkLabel(C_guide_frame,
                             text="C",
                             font=ABCs_font)
    # Place the widget to the grid
    C_label.grid(row=0, column=1)

    # Create guide label
    C_guide_label = ctk.CTkLabel(C_guide_frame,
                                 text="Now you have clusters. In next step, try to identify what is the characteristic of all images in each cluster and let the machine learn your names and predict which cluster new images would fall into. Did the machine get it right?"
                                      "\nWhen done, click on 'Next'.",
                                 wraplength=500)
    # Place the guide label to the grid
    C_guide_label.grid(row=9, column=1)

    # Define a button which goes to the classification page
    go_to_classification_page_button = ctk.CTkButton(C_guide_frame,
                                       text="Continue",
                                       command=classification_page)
    # Place the button to the grid
    go_to_classification_page_button.grid(row=10, column=1)

# Define the image clustering page
def clustering_page():
    global main_frame, number_of_clusters_frame

    # Destroy previous widgets
    destroy_previous_widgets()

    # Create main frame for subframes to use
    main_frame = create_main_frame()

    # Define a frame for the clustering page
    number_of_clusters_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
    # Place the frame for the clustering page
    number_of_clusters_frame.grid(row=0, column=1)

    # Define the grid layout for the clustering page frame
    for i in range(15):
        number_of_clusters_frame.rowconfigure(i, weight=1)
    for i in range(5):
        number_of_clusters_frame.columnconfigure(i, weight=1)

    # Define function to display the current state of slider
    def slider_clusters_behavior(slider_clusters_value):
        # Change the text of slider value indicator to the current state of the slider
        slider_clusters_value_label.configure(text=f"Number of clusters: {round(slider_clusters_value)}")

    # Define a slider
    slider_clusters = ctk.CTkSlider(number_of_clusters_frame, from_=3, to=9, command = slider_clusters_behavior)
    # Place it to the number of clusters section
    slider_clusters.grid(row=11, column=2)

    # Define a label which shows how many clusters are currently selected through slider
    slider_clusters_value_label = ctk.CTkLabel(number_of_clusters_frame, text=f"Number of clusters: {round(slider_clusters.get())}")
    # Place the slider to the number of clusters frame
    slider_clusters_value_label.grid(row=12, column=2)

    # Call a function that displays the joined dataset from the picture completion page
    display_joined_dataset()

    # Define the clustering machine learning algorithm process which will happen when button Cluster will be clicked
    def cluster_faces():
        # Make these variables available across functions
        global today_input_samples, renaissance_input_samples, slider_reclusters, clusters_frame, filenames_of_clusters,\
            data_file_paths, Z

        # For all but the first time, get the info about how many clusters to have from the recluster slider
        try:
            # Get the number of clusters to be included in the dataset from the slider
            n_clusters = round(slider_reclusters.get())
        # Only for the first clustering, ge the info about how many cluster to have from the cluster slider
        except:
            # Get the number of clusters to be included in the dataset from the slider
            n_clusters = round(slider_clusters.get())


        # Create a list into which dataset will be put
        data = []

        # Create a list into which file paths will be put
        data_file_paths = []

        # For every file in the renaissance samples
        for file in renaissance_input_samples:
            # Specify the file path
            img_path = os.path.join(input_dir_renaissance, file)
            # Load the image
            img = Image.open(img_path)
            # Turn the image to grayscale
            img = ImageOps.grayscale(img)
            # Turn the image into array (2D)
            img = np.asarray(img)
            # Make the 2D array 1D array
            img = img.flatten()
            # Append the array to the data list
            data.append(img)
            # Append the file path to the data_file_paths list
            data_file_paths.append(img_path)

        # For every file in today's samples
        for file in today_input_samples:
            # Specify the file path
            img_path = os.path.join(input_dir_today, file)
            # Load the image
            img = Image.open(img_path)
            # Turn the image to grayscale
            img = ImageOps.grayscale(img)
            # Turn the image into array (2D)
            img = np.asarray(img)
            # Make the 2D array 1D array
            img = img.flatten()
            # Append the array to the data list
            data.append(img)
            # Append the file path to the data_file_paths list
            data_file_paths.append(img_path)

        # Pair images and their file paths together in a list so after reshuffling,
        # image will still have the file path in the same position in the other list as the image itself
        list_for_shuffling = list(zip(data, data_file_paths))

        # Shuffle the paired images with their corresponding file paths
        random.shuffle(list_for_shuffling)

        # Update the original two lists based on the paired reshuffled order
        data, data_file_paths = zip(*list_for_shuffling)

        # Turn the data into an array
        data = np.asarray(data)

        # Obtain the initial centroids, so the obtained results are repeatable
        np.random.seed(1)

        # Create the clustering object
        kmeans = KMeans(n_clusters=n_clusters, init='random')
        # Train the clustering object model
        kmeans.fit(data)
        # Get the clusters
        Z = kmeans.predict(data)

        print(Z)

        # Define a list to store the filenames of generated clusters
        filenames_of_clusters = []

        # For every cluster
        for i in range(0, n_clusters):
            # Row in Z for elements of cluster i
            row = np.where(Z == i)[0]
            # Number of elements for each cluster
            num = row.shape[0]
            # Number of rows in the figure of the cluster
            r = np.floor(num / 3)

            print("cluster " + str(i))
            print(str(num) + " elements")

            # Make number of rows an integer for matplotlib to be able to generate the plots
            r = int(r)

            # Define figure size
            plt.figure(figsize=(10, 10))
            # For every cluster
            for k in range(0, num):
                plt.subplot(r + 1, 3, k + 1)  # plt.subplot(r+1, 10, k+1)
                # Get the image from the array
                image = data[row[k],]
                # Turn the image pixels as array back to a 64x64 picture
                image = image.reshape(64, 64)
                # Display th images as grayscale
                plt.imshow(image, cmap='gray')
                # Do not show axis
                plt.axis('off')
            # Dynamically define the file name of each cluster based on its number
            name = f"Cluster {i}.png"
            # Append this file name to the list of file names for future storage
            filenames_of_clusters.append(name)
            # Remove non-needed space around the images in a cluster
            plt.tight_layout()
            # Save the figure as png
            plt.savefig(name, bbox_inches="tight")
            # Load the saved image
            image = Image.open(name)
            # Add a red border around the image
            image_with_border = ImageOps.expand(image, border=2, fill="red")
            # Save the image again
            image_with_border.save(name)

        # Delete the cluster frame in case of reclustering so newly-generated clusters can be displayed
        try:
            clusters_frame.destroy()
        # In the case of first generation of clusters, delete the initial clustering page
        except:
            number_of_clusters_frame.destroy()

        # Define a frame for the view of clusters
        clusters_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
        # Place the frame for the view of clusters
        clusters_frame.grid(row=0, column=1)

        # Define the grid layout for the view of clusters
        for i in range(6):
            clusters_frame.rowconfigure(i, weight=1)
        for i in range(3):
            clusters_frame.columnconfigure(i, weight=1)

        # Display the clusters
        # Define the initial cell
        row = 0
        column = 0
        # For every cluster
        for cluster in filenames_of_clusters:
            # Specify the file path
            img_path = cluster
            # Open the image in the gui
            image = ctk.CTkImage(light_image=Image.open(img_path), size=(300, 300))
            # Create a widget of that image
            image_label = ctk.CTkLabel(clusters_frame, image=image, text="")
            # Place that image to the next cell
            image_label.grid(row=row, column=column)
            # If there are columns still left in the row
            if column < 4:
                # Make the next image be in the next free column of that row
                column += 1
            # If there are no more free columns int the row
            elif column == 4:
                # Move to the next row
                row += 1
                column = 0

            # Define function to display the current state of slider
            def slider_reclusters_behavior(slider_reclusters_value):
                # Change the text of slider value indicator to the current state of the slider
                slider_reclusters_value_label.configure(text=f"Number of clusters: {round(slider_reclusters_value)}")

            # Define a slider for reclustering
            slider_reclusters = ctk.CTkSlider(clusters_frame, from_=3, to=9, command=slider_reclusters_behavior)
            # Place it to the clusters section
            slider_reclusters.grid(row=3, column=0, columnspan=5)

            # Define a label which shows how many clusters are currently selected through slider
            slider_reclusters_value_label = ctk.CTkLabel(clusters_frame,
                                                       text=f"Number of clusters: {round(slider_reclusters.get())}")
            # Place the slider to the clusters frame
            slider_reclusters_value_label.grid(row=4, column=0, columnspan=5)

            # Define a button which starts the reclustering
            recluster_button = ctk.CTkButton(clusters_frame,
                                           text="Recluster",
                                           command=cluster_faces)
            # Place the button in the clusters section
            recluster_button.grid(row=5, column=0, columnspan=5)

            # Define a button for the next page
            go_to_classification_button = ctk.CTkButton(clusters_frame,
                                             text="Next",
                                             command=C_guide)
            # Place the button in the clusters section
            go_to_classification_button.grid(row=6, column=0, columnspan=5)




    # Define a button which starts the clustering
    cluster_button = ctk.CTkButton(number_of_clusters_frame,
                               text="Cluster",
                                   command=cluster_faces)
    # Place the button in the number of clusters section
    cluster_button.grid(row=14, column=2)


# Define the B guide page
def B_guide():
    # Make the main frame available to use for the B guide page
    global main_frame

    # Destroy previous widgets
    destroy_previous_widgets()

    # Create main frame for subframes to use
    main_frame = create_main_frame()

    # Define a frame for B guide page
    B_guide_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
    # Place the A guide page frame
    B_guide_frame.grid(row=0, column=1)

    # Define the grid layout for A guide page
    for i in range(10):
        B_guide_frame.rowconfigure(i, weight=1)
    B_guide_frame.columnconfigure(0, weight=1)

    # Create the A text widget
    B_label = ctk.CTkLabel(B_guide_frame,
                             text="B",
                             font=ABCs_font)
    # Place the widget to the grid
    B_label.grid(row=0, column=1)

    # Create guide label
    B_guide_label = ctk.CTkLabel(B_guide_frame,
                                 text="Now that you have your dataset, let's use these mixed pictures to let the machine separate them into clusters of similar images."
                                      "\nYou can choose how many clusters do you want the machine to create and observe how the content of clusters change with different cluster numbers.\nFeel free to recluster to explore more.\nWhen done, click on 'Next'.",
                                 wraplength=500)
    # Place the guide label to the grid
    B_guide_label.grid(row=9, column=1)

    # Define a button which goes to the percentage page
    go_to_clustering_page_button = ctk.CTkButton(B_guide_frame,
                                       text="Continue",
                                       command=clustering_page)
    # Place the button to the grid
    go_to_clustering_page_button.grid(row=10, column=1)


# Define the image completion (choose percentage) page
def choose_percentage_page():
    global main_frame

    # Destroy previous widgets
    destroy_previous_widgets()

    # Create main frame for subframes to use
    main_frame = create_main_frame()
    
    # Define a frame for the choose percentage and results display section
    choose_percentage_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
    # Place the choose percentage section to the main frame
    choose_percentage_frame.grid(row=0, column=1)
    
    # Define the grid layout for the choose percentage section
    for i in range(5):
        choose_percentage_frame.rowconfigure(i, weight=1)
    choose_percentage_frame.columnconfigure(0, weight=1)

    # Define a slider
    slider = ctk.CTkSlider(choose_percentage_frame, from_=0, to=100)
    # Place it to the choose percentage and results display section
    slider.grid(row=2, column=0)

    # Define the face completion machine learning algorithm process which will happen when button Mix will be clicked
    def mix_faces():
        global today_input_samples, renaissance_input_samples

        # Get the percentage of renaissance faces to be included in the dataset from the slider
        percentage_renaissance = slider.get()

        # Create a list into which dataset will be put
        data = []

        # Create a list into which photos to be predicted upon will be put
        input_data = []

        # Get an integer of how many renaissance images to include (max 50)
        n_samples_renaissance = round((percentage_renaissance / 2))

        # Get an integer of how many today images to include (the remaining count to 50)
        n_samples_today = 50 - n_samples_renaissance

        # Randomly get that many percent of renaissance samples from the dataset
        renaissance_input_samples = random.sample(renaissance_dataset, n_samples_renaissance)

        # Randomly get that many percent of renaissance samples from the dataset
        today_input_samples = random.sample(today_dataset, n_samples_today)

        # TRAINING DATA

        # For every file in the renaissance samples
        for file in renaissance_input_samples:
            # Specify the file path
            img_path = os.path.join(input_dir_renaissance, file)
            # Load the image
            img = Image.open(img_path)
            # Turn the image to grayscale
            img = ImageOps.grayscale(img)
            # Turn the image into array (2D)
            img = np.asarray(img)
            # Make the 2D array 1D array
            img = img.flatten()
            # Append the array to the data list
            data.append(img)

        # For every file in today's samples
        for file in today_input_samples:
            # Specify the file path
            img_path = os.path.join(input_dir_today, file)
            # Load the image
            img = Image.open(img_path)
            # Turn the image to grayscale
            img = ImageOps.grayscale(img)
            # Turn the image into array (2D)
            img = np.asarray(img)
            # Make the 2D array 1D array
            img = img.flatten()
            # Append the array to the data list
            data.append(img)

        # Reshuffle the data list to avoid bias when training the data
        random.shuffle(data)

        # Turn the data into an array
        data = np.asarray(data)

        # Define how many pixels each image has (64x64 = 4096)
        n_pixels = 4096

        # Get the upper half of the faces
        X_train = data[:, : (n_pixels + 1) // 2]
        # Get the lower half of the faces
        y_train = data[:, n_pixels // 2:]

        # Create an estimator object
        estimator = KNeighborsRegressor(n_neighbors=2)
        #estimator = DecisionTreeRegressor(max_depth=10, random_state=0)
        #estimator = ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0)

        # Train the estimator so that it learns a model of the data
        estimator.fit(X_train, y_train)

        # TESTING DATA

        # For every file in today's samples
        for file in os.listdir(input_dir_today):
            # Specify the file path
            img_path = os.path.join(input_dir_today, file)
            # Load the image
            img = Image.open(img_path)
            # Turn the image to grayscale
            img = ImageOps.grayscale(img)
            # Turn the image into array (2D)
            img = np.asarray(img)
            # Make the 2D array 1D array
            img = img.flatten()
            # Append the array to the data list
            input_data.append(img)

        # Reshuffle the data list to show different results
        random.shuffle(input_data)

        # Turn the data into an array
        input_data = np.asarray(input_data)

        # Get the upper part of the faces
        X_test = input_data[:, : (n_pixels + 1) // 2]
        # Get the lower part of the faces
        y_test = input_data[:, n_pixels // 2:]

        # Predict the lower part of the faces based on the upper part
        y_test_predict = estimator.predict(X_test)

        print(X_train.shape)
        print(y_train.shape)

        # How many faces to display
        n_faces = 6

        # Plot the completed faces
        image_shape = (64, 64)

        n_cols = 2

        # Check the performance of the estimator on testing faces:

        plt.figure(figsize=(2.0 * n_cols, 2.26 * n_faces))
        plt.suptitle("Face completion with knn regression: \ntesting set", size=16)

        for i in range(n_faces):
            true_face = np.hstack((X_test[i], y_test[i]))

            if i:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
            else:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title="true faces")

            sub.axis("off")
            sub.imshow(
                true_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest"
            )

            completed_face = np.hstack((X_test[i], y_test_predict[i]))
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2)

            sub.axis("off")
            sub.imshow(
                completed_face.reshape(image_shape),
                cmap=plt.cm.gray,
                interpolation="nearest",
            )
        # Save the result as image to display in GUI
        plt.savefig('face_completions_result.png')

        # Locate the result image
        result_path = "face_completions_result.png"
        # Load it for the GUI
        image = ctk.CTkImage(light_image=Image.open(result_path), size=(160, 542))
        # Create an image widget to display in GUI
        image_label = ctk.CTkLabel(choose_percentage_frame, image=image, text="")
        # Place the image in the choose percentage and display results section
        image_label.grid(row=4, column=0)

        # Destroy the previously shown today dataset
        todays_dataset_frame.destroy()
        # Show the image from today's dataset which were used in the current machine learning
        display_todays_dataset(today_input_samples,input_dir_today)

        # Destroy the previously shown renaissance dataset
        renaissance_dataset_frame.destroy()
        # Show the image from renaissance dataset which were used in the current machine learning
        display_renaissance_dataset(renaissance_input_samples, input_dir_renaissance)

        # Define a button which goes to the next page
        go_to_clustering_button = ctk.CTkButton(choose_percentage_frame,
                                   text="Next",
                                   command=B_guide)
        # Place the button in the choose percentage and display results section
        go_to_clustering_button.grid(row=5, column=0)

    # Define a button which starts the face completion process
    mix_button = ctk.CTkButton(choose_percentage_frame,
                                            text="Mix",
                               command=mix_faces)
    # Place the button in the choose percentage and display results section
    mix_button.grid(row=3, column=0)

    # Display the original full today dataset when loading the page for the first time
    display_todays_dataset(today_dataset, input_dir_today)

    # Display the original full renaissance dataset when loading the page for the first time
    display_renaissance_dataset(renaissance_dataset, input_dir_renaissance)

# Define the A guide page
def A_guide():
    # Make the main frame available to use for the A guide page
    global main_frame

    # Destroy previous widgets
    destroy_previous_widgets()

    # Create main frame for subframes to use
    main_frame = create_main_frame()

    # Define a frame for A guide page
    A_guide_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
    # Place the A guide page frame
    A_guide_frame.grid(row=0, column=1)

    # Define the grid layout for A guide page
    for i in range(10):
        A_guide_frame.rowconfigure(i, weight=1)
    A_guide_frame.columnconfigure(0, weight=1)

    # Create the A text widget
    A_label = ctk.CTkLabel(A_guide_frame,
                             text="A",
                             font=ABCs_font)
    # Place the widget to the grid
    A_label.grid(row=0, column=1)

    # Create guide label
    A_guide_label = ctk.CTkLabel(A_guide_frame,
                                 text="In the beginning, let machine learning generate lower part of contemporary faces.\nThese lower halves will be a blend of faces from today's people and Reinassence people.\nYou can choose how much the second half will have of today's and Renaissance features by changing the amount of pictures from these two periods in a training dataset for the machine.\nUse slider to change the mix and feel free to remix again to observe patterns!\nWhen done, click on 'Next'.",
                                 wraplength=500)
    # Place the guide label to the grid
    A_guide_label.grid(row=9, column=1)

    # Define a button which goes to the percentage page
    go_to_choose_percentage_page_button = ctk.CTkButton(A_guide_frame,
                                       text="Continue",
                                       command=choose_percentage_page)
    # Place the button to the grid
    go_to_choose_percentage_page_button.grid(row=10, column=1)


# Define the welcome page
def welcome_page():
    # Make the main frame available to use for the welcome page
    global main_frame

    # Destroy previous widgets
    destroy_previous_widgets()

    # Create main frame for subframes to use
    main_frame = create_main_frame()

    # Define a frame for the welcome page
    welcome_page_frame = ctk.CTkFrame(main_frame, fg_color=background_color)
    # Place the welcome page to the grid
    welcome_page_frame.grid(row=0, column=1)

    # Define the grid layout for the welcome page
    for i in range(10):
        welcome_page_frame.rowconfigure(i, weight=1)
    welcome_page_frame.columnconfigure(0, weight=1)

    # Create the ABC label
    ABC_label = ctk.CTkLabel(welcome_page_frame,
                                 text="ABC",
                                 font=ABCs_font)
    # Place it to the gird
    ABC_label.grid(row=0, column=1)

    # Create label with the welcome text
    welcome_label = ctk.CTkLabel(welcome_page_frame, text="Welcome to ABCs of Machine Learning \n Play with some methods of machine learning and explore how it works. \n", wraplength=500)
    # Place it to the grid
    welcome_label.grid(row=9, column=1)

    # Define a button which introduces the A part
    get_started_button = ctk.CTkButton(welcome_page_frame,
                                         text="Get started",
                                         command=A_guide)
    # Place the button to the grid
    get_started_button.grid(row=10, column=1)


# Display the welcome page when the user opens the interface
welcome_page()

# Run the app
window.mainloop()