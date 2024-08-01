import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
import inspect

def load_train(path):

    labels = pd.read_csv(path + 'labels.csv')
    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255,
        horizontal_flip=True
    )
    train_gen_flow=train_datagen.flow_from_dataframe(
    dataframe=labels,
    directory= path + 'final_files/',
    target_size=(224,224),
    batch_size=16,
    class_mode='raw',
    x_col='file_name',
    y_col='real_age',
    subset='training',
    seed=12345
)


    return train_gen_flow
    
    
def load_test(path):
    
    labels = pd.read_csv(path + 'labels.csv')
    test_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255
    )
    test_gen_flow=test_datagen.flow_from_dataframe(
    dataframe=labels,
    directory=path + 'final_files/',
    target_size=(224,224),
    batch_size=16,
    class_mode='raw',
    x_col='file_name',
    y_col='real_age',
    subset='validation',
    seed=12345
)

    return test_gen_flow
    
def create_model(input_shape):

    backbone=ResNet50(input_shape=input_shape,weights='imagenet',include_top=False)
    model=Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=1,activation='relu'))
    optimizer=Adam(lr=0.0005)
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])

    return model
    
def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data) 
    model.fit(train_data,
              validation_data=test_data,
              epochs=epochs,
              batch_size=batch_size,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
             verbose=2
             )
    

    return model

def string():
    init_str = """
    import pandas as pd

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.resnet import ResNet50
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
    from tensorflow.keras.optimizers import Adam

    def load_train(path):

        labels = pd.read_csv(path + 'labels.csv')
        train_datagen = ImageDataGenerator(
            validation_split=0.25,
            rescale=1./255,
            horizontal_flip=True
        )
        train_gen_flow=train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory= path + 'final_files/',
        target_size=(224,224),
        batch_size=16,
        class_mode='raw',
        x_col='file_name',
        y_col='real_age',
        subset='training',
        seed=12345
    )


        return train_gen_flow


    def load_test(path):

        labels = pd.read_csv(path + 'labels.csv')
        test_datagen = ImageDataGenerator(
            validation_split=0.25,
            rescale=1./255
        )
        test_gen_flow=test_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'final_files/',
        target_size=(224,224),
        batch_size=16,
        class_mode='raw',
        x_col='file_name',
        y_col='real_age',
        subset='validation',
        seed=12345
    )

        return test_gen_flow

    def create_model(input_shape):

        backbone=ResNet50(input_shape=input_shape,weights='imagenet',include_top=False)
        model=Sequential()
        model.add(backbone)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(units=1,activation='relu'))
        optimizer=Adam(lr=0.0005)
        model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])

        return model

    def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                    steps_per_epoch=None, validation_steps=None):

        
        if steps_per_epoch is None:
            steps_per_epoch = len(train_data)
        if validation_steps is None:
            validation_steps = len(test_data) 
        model.fit(train_data,
                validation_data=test_data,
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                verbose=2
                )
        return model
    """
    
    path='./src/models/'
    
    with open(path+'run_model_on_gpu.py', 'w') as f:
        f.write(init_str)
        f.write('\n\n')
        for fn_name in [load_train, load_test, create_model, train_model]:
            src = inspect.getsource(fn_name)
            f.write(src)
            f.write('\n\n')