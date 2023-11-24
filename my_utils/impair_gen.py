from tensorflow.keras.preprocessing.image import ImageDataGenerator


def impairGenerator(input_path,target_path,nx,ny,batch_size,seed):
    
    impair_gen_args = dict(horizontal_flip=True,vertical_flip=True,fill_mode='reflect')
    
    input_datagen = ImageDataGenerator(**impair_gen_args)
    target_datagen = ImageDataGenerator(**impair_gen_args)
    
    input_generator = input_datagen.flow_from_directory(
        input_path,
        class_mode = None,
        color_mode = 'grayscale',
        target_size=(ny,nx),
        batch_size = batch_size,
        seed = seed)
    
    target_generator = target_datagen.flow_from_directory(
        target_path,
        class_mode = None,
        color_mode = 'grayscale',
        target_size=(ny,nx),
        batch_size = batch_size,
        seed = seed)
    
    impair_generator = zip(input_generator,target_generator)
    
    for (iinput, target) in impair_generator:
        iinput, target = preprocess_data(iinput, target)
        yield (iinput, target)
        
        
def preprocess_data(iinput,target):
    iinput = iinput / 65536.
    target = target / 65536. 
    return (iinput, target)