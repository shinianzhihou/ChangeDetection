import random
import math
import numpy as np 
from PIL import  Image
from torchvision.transforms.functional import  to_pil_image,to_tensor
import cv2

def RandomCrop(p, cropWidth, cropHeight, centre=False):
    '''
        按长宽从原图随机裁剪小块图片
    '''

    def do(images):
        
        w, h = images[0].size  #读出图片长宽

        #如果裁剪尺度比本身尺度大 返回原图
        if cropWidth > w or cropHeight > h:
            return images

        #生成锚点
        left_shift = random.randint(0, int((w - cropWidth)))
        down_shift = random.randint(0, int((h - cropHeight)))
        
        #按概率操作
        if random.random() < p:
            aug_imgs  = [] 
            for image in images:
                item = image.crop(((w/2)-(cropWidth/2), (h/2)-(cropHeight/2), (w/2)+(cropWidth/2), (h/2)+(cropHeight/2))) if centre \
                     else image.crop((left_shift, down_shift, cropWidth + left_shift, cropHeight + down_shift))
                aug_imgs.append(item) 
        else:
            aug_imgs = images
        
        return aug_imgs

    return do


def RandomRotate90(p):
    '''
        图片随机转n*90度 (n in [1,2,3])
    '''
    def do(images):
        if random.random()<p:
            factor = random.randint(1,3)
            aug_imgs = []
            for image in images:
                aug_imgs.append(image.rotate(90 * factor, expand=True))
        
        else:
            aug_imgs = images
        
        return aug_imgs
    
    return do 


def RandomVerticalFlip(p):
    '''
        随机 垂直 翻转
    '''
    def do(images):

        if random.random() < p:
            aug_imgs = []
            for image in images:
                item = image.transpose(Image.FLIP_TOP_BOTTOM)
                aug_imgs.append(item)
        else:
            aug_imgs = images
        

        return aug_imgs
    
    return do



def RandomHorizontalFlip(p):
    '''
        随机 水平 翻转
    '''
    def do(images):
              
        if random.random() < p:
            aug_imgs = []
            for image in images:
                item = image.transpose(Image.FLIP_LEFT_RIGHT)
                aug_imgs.append(item)
        else:
            aug_imgs = images
        

        return aug_imgs
    
    return do




def RandomFlip(p):
    '''
        随机 水平或垂直 翻转
    '''
    def do(images):
        
        if random.random() < p:
            axis = random.randint(1, 2)
            aug_imgs = []

            ###2种翻转情况###
            if axis == 1:
                for image in images:
                    item = image.transpose(Image.FLIP_LEFT_RIGHT)
                    aug_imgs.append(item)

            else:
                for image in images:
                    item = image.transpose(Image.FLIP_TOP_BOTTOM)
                    aug_imgs.append(item)
            

            #############
            
        else:
            aug_imgs = images
        

        return aug_imgs
    
    return do
            




def ToPILImage(mode=None):
    '''
        转pil
    '''
    def do(images):
        images = [F.to_pil_image(img,mode) for img in images]
        return images

    return do 


def ToTensor(mode=None):
    '''
        转torch tensor
    '''
    def do(images):
        images = [F.to_tensor(img,mode) for img in images]
        return images

    return do 



def ToNumpy(scale_factor=None):
    '''
        包括以下步骤：
        1.转numpy 
        2.调整图片大小 
        3.归一化 
        4.标签二值化（主要针对有压缩的图片jpg等） 
        5.调整为channel first
        （转torch tensor的过程 交给dataloader）
    '''


    def do(images): 


        images = [np.array(img) for img in images] 

        ##################
        '''
            在测试图片时，经过上采样和下采样后
            若图片长宽不是2^n的倍数
            长宽会改变 
            skip connection的过程会报错 
            所以 事先将图片长宽调整为scale_factor的倍数 
            在测试阶段预测完成后 
            根据标签长宽再将预测结果的长宽调整回来
        '''
        if scale_factor:
            h,w = images[0].shape[0],images[1].shape[1]        
            if h%scale_factor!=0 or w%scale_factor!=0:
                h_new = int(h/scale_factor + 0.5)*scale_factor
                w_new = int(w/scale_factor + 0.5)*scale_factor

                images[0] = cv2.resize(images[0],(w_new,h_new))
                images[1] = cv2.resize(images[1],(w_new,h_new))
        ###################
        images = [img/255. for img in images] 
        images[-1] = images[-1][...,None]   #升维 1通道
        images[-1][images[-1]>0.5] = 1.
        images[-1][images[-1]<=0.5] = 0. 
        images = [img.transpose(2,0,1) for img in images]
        return images
    
    return do 

               


class RandomRotation():
    """
    参考：https://github.com/mdbloice/Augmentor
    This class is used to perform rotations on images by arbitrary numbers of
    degrees.

    Images are rotated **in place** and an image of the same size is
    returned by this function. That is to say, that after a rotation
    has been performed, the largest possible area of the same aspect ratio
    of the original image is cropped from the skewed image, and this is
    then resized to match the original image size.

    The method by which this is performed is described as follows:

    .. math::

        E = \\frac{\\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}}\\Big(X-\\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}} Y\\Big)}{1-\\frac{(\\sin{\\theta_{a}})^2}{(\\sin{\\theta_{b}})^2}}

    which describes how :math:`E` is derived, and then follows
    :math:`B = Y - E` and :math:`A = \\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}} B`.

    The :ref:`rotating` section describes this in detail and has example
    images to demonstrate this.
    """
    def __init__(self, p, max_left_rotation, max_right_rotation):
        """
        As well as the required :attr:`probability` parameter, the
        :attr:`max_left_rotation` parameter controls the maximum number of
        degrees by which to rotate to the left, while the
        :attr:`max_right_rotation` controls the maximum number of degrees to
        rotate to the right.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param max_left_rotation: The maximum number of degrees to rotate
         the image anti-clockwise.
        :param max_right_rotation: The maximum number of degrees to rotate
         the image clockwise.
        :type probability: Float
        :type max_left_rotation: Integer
        :type max_right_rotation: Integer
        """
        self.p = p
        self.max_left_rotation = -abs(max_left_rotation)   # Ensure always negative
        self.max_right_rotation = abs(max_right_rotation)  # Ensure always positive

    def __call__(self, images):
        """
        Perform the rotation on the passed :attr:`image` and return
        the transformed image. Uses the :attr:`max_left_rotation` and
        :attr:`max_right_rotation` passed into the constructor to control
        the amount of degrees to rotate by. Whether the image is rotated
        clockwise or anti-clockwise is chosen at random.

        :param images: The image(s) to rotate.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        # TODO: Small rotations of 1 or 2 degrees can create black pixels
        random_left = random.randint(self.max_left_rotation, 0)
        random_right = random.randint(0, self.max_right_rotation)

        left_or_right = random.randint(0, 1)

        rotation = 0

        if left_or_right == 0:
            rotation = random_left
        elif left_or_right == 1:
            rotation = random_right

        def do(image):
            # Get size before we rotate
            x = image.size[0]
            y = image.size[1]

            # Rotate, while expanding the canvas size
            image = image.rotate(rotation, expand=True, resample=Image.BICUBIC)

            # Get size after rotation, which includes the empty space
            X = image.size[0]
            Y = image.size[1]

            # Get our two angles needed for the calculation of the largest area
            angle_a = abs(rotation)
            angle_b = 90 - angle_a

            # Python deals in radians so get our radians
            angle_a_rad = math.radians(angle_a)
            angle_b_rad = math.radians(angle_b)

            # Calculate the sins
            angle_a_sin = math.sin(angle_a_rad)
            angle_b_sin = math.sin(angle_b_rad)

            # Find the maximum area of the rectangle that could be cropped
            E = (math.sin(angle_a_rad)) / (math.sin(angle_b_rad)) * \
                (Y - X * (math.sin(angle_a_rad) / math.sin(angle_b_rad)))
            E = E / 1 - (math.sin(angle_a_rad) ** 2 / math.sin(angle_b_rad) ** 2)
            B = X - E
            A = (math.sin(angle_a_rad) / math.sin(angle_b_rad)) * B

            # Crop this area from the rotated image
            # image = image.crop((E, A, X - E, Y - A))
            image = image.crop((int(round(E)), int(round(A)), int(round(X - E)), int(round(Y - A))))

            # Return the image, re-sized to the size of the image passed originally
            return image.resize((x, y), resample=Image.BICUBIC)


        if random.random() < self.p:
            augmented_images = []
            for image in images:
                augmented_images.append(do(image))
        else:
            augmented_images = images

        return augmented_images




class Zoom():
    """
    参考：https://github.com/mdbloice/Augmentor
    This class is used to enlarge images (to zoom) but to return a cropped
    region of the zoomed image of the same size as the original image.
    """
    def __init__(self, probability, min_factor, max_factor):
        """
        The amount of zoom applied is randomised, from between
        :attr:`min_factor` and :attr:`max_factor`. Set these both to the same
        value to always zoom by a constant factor.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param min_factor: The minimum amount of zoom to apply. Set both the
         :attr:`min_factor` and :attr:`min_factor` to the same values to zoom
         by a constant factor.
        :param max_factor: The maximum amount of zoom to apply. Set both the
         :attr:`min_factor` and :attr:`min_factor` to the same values to zoom
         by a constant factor.
        :type probability: Float
        :type min_factor: Float
        :type max_factor: Float
        """
        self.p = probability
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, images):
        """
        Zooms/scales the passed image(s) and returns the new image.

        :param images: The image(s) to be zoomed.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        factor = round(random.uniform(self.min_factor, self.max_factor), 2)

        def do(image):
            w, h = image.size

            image_zoomed = image.resize((int(round(image.size[0] * factor)),
                                         int(round(image.size[1] * factor))),
                                         resample=Image.BICUBIC)
            w_zoomed, h_zoomed = image_zoomed.size

            return image_zoomed.crop((math.floor((float(w_zoomed) / 2) - (float(w) / 2)),
                                      math.floor((float(h_zoomed) / 2) - (float(h) / 2)),
                                      math.floor((float(w_zoomed) / 2) + (float(w) / 2)),
                                      math.floor((float(h_zoomed) / 2) + (float(h) / 2))))

        if random.random() < self.p:
            augmented_images = []
            for image in images:
                augmented_images.append(do(image))
        else:
            augmented_images = images

        return augmented_images
