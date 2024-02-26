library(tensorflow)
library(keras)

set.seed(42)

train_dir <- "C:\\Users\\Siddharth gupta\\Downloads\\training_data"
val_dir <- "C:\\Users\\Siddharth gupta\\Downloads\\validation_data"

train_datagen <- image_data_generator(rescale=1/255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=TRUE)

val_datagen <- image_data_generator(rescale=1/255)

train_generator <- flow_images_from_directory(train_dir,
                                              generator=train_datagen,
                                              target_size=c(150,150),
                                              batch_size=20,
                                              class_mode="categorical")

val_generator <- flow_images_from_directory(val_dir,
                                            generator=val_datagen,
                                            target_size=c(150,150),
                                            batch_size=20,
                                            class_mode="categorical")


model <- keras_model_sequential() %>%
  layer_conv_2d(filters=32, kernel_size=c(3,3), activation="relu", input_shape=c(150,150,3)) %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=64, kernel_size=c(3,3), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=128, kernel_size=c(3,3), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=128, kernel_size=c(3,3), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units=512, activation="relu") %>%
  layer_dropout(rate=0.5) %>%
  layer_dense(units=7, activation="softmax")

model

model %>% compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=c("accuracy"))


history <- model %>% fit(
  train_generator,
  steps_per_epoch=13,
  epochs=7,
  validation_data=val_generator,
  validation_steps=50
)

plot(history)

model %>% save_model_hdf5("potato_disease_model.h5")

model <- load_model_hdf5("potato_disease_model.h5")

image_path <- "C:\\Users\\Siddharth gupta\\Downloads\\validation_data\\Pink Rot\\37.jpg"
image <- image_load(image_path, target_size = c(150, 150))
image_array <- image_to_array(image)
image_array <- array_reshape(image_array, c(1, dim(image_array)))

prediction <- predict(model, image_array)

class_labels <- c("Pink rot", "Miscellaneous", "Healthy Potato", "Dry Rot", "Common Scab", "Black Leg", "Black Scurf")
class_index <- which.max(prediction)
class_label <- class_labels[class_index]

class_label

