import os

# specify example image paths: 
image_paths =  [
    'synpic50962.jpg',
    'synpic52767.jpg',
    'synpic30324.jpg',
    'synpic21044.jpg',
    'synpic54802.jpg',
    'synpic57813.jpg',
    'synpic47964.jpg'
]
image_paths = [os.path.join('../img', p) for p in image_paths]

    

def clean_generation(response):
    """
    for some reason, the open-flamingo based model slightly changes the input prompt (e.g. prepends <unk>, an adds some spaces)
    """
    return response.replace('<unk> ', '').strip()

