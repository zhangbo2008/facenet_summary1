
'''
这个文件用于拿一张图片,跟库里面的图片进行比较.返回是否在库中存在,如果存在就
返回库中文件的名字,否则返回unknown

'''









from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument('--model', type=str,required='False',
    #                     help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    # parser.add_argument('--image_files', required='False',type=str, nargs='+', help='Images to compare')
    # parser.add_argument('--image_size', type=int,required='False',
    #                     help='Image size (height, width) in pixels.', default=160)
    # parser.add_argument('--margin', type=int,required='False',
    #                     help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    # parser.add_argument('--gpu_memory_fraction', type=float,required='False',
    #                     help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    '''
    超参数都写这里面.
    '''


















    return parser.parse_args(argv)
def main(args):
    args.model='model/20180402-114759'



    args.image_files='data/images'

    args.input='tmp.png'
    args.image_size=160 #不是输入图片的大小.而是处理大小,不用动,给160就行.
    args.gpu_memory_fraction=1.0
    args.margin=44


    import os,sys
    args.image_files=    [args.image_files+'/'+i for i in os.listdir(args.image_files)]




    import sys,os
    os.system('cp tmp.png data/images')
    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction) #  3,160,160,3
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            #得到的emb就是向量.
            nrof_images = len(args.image_files)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, args.image_files[i]))
            print('')
            
            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    print('  %1.4f  ' % dist, end='')
                print('')
            print('下面打印上面结果的说明')
            print('首先是编码和图片名称的对应,编码从0开始:',args.image_files)
            print('如果两个图片距离小于1.1就说明是同一人!这个1.1是论文里面给的系数,鲁棒性比较高.')

            
            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor) #这里面用的是mtcnn算法.已经封装过docker了,看之前docker可以学习.#bounding _boxes:前4个坐标是box坐标,最后一个是score
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')#把人脸切出来,然后resize成160
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    os.system('rm -f   data/images/tmp.png')
    print("都over了!")
