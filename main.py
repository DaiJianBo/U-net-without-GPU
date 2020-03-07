from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
#Keras中，当数据比较大时，不能全部载入内存，在训练的时候就需要利用train_on_batch或fit_generator进行训练了

model = unet()   #载入模型
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#该回调函数将在每个epoch后保存模型到filepath
#使用keras搭建模型，训练时验证集上val_acc达到1了，但在测试数据集上面模型还没有完全收敛。
# 由于在ModelCheckpoint的参数设置时设置了仅保留最佳模型，导致无法保存最新的更好的模型。

model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
# 通过Python generator产生一批批的数据用于训练模型。generator可以和模型并行运行，例如，可以使用CPU生成批数据同时在GPU上训练模型
# generator：一个generator或Sequence实例，为了避免在使用multiprocessing时直接复制数据。
# steps_per_epoch：从generator产生的步骤的总数（样本批次总数）。通常情况下，应该等于数据集的样本数量除以批量的大小。
# epochs：整数，在数据集上迭代的总数。
# works：在使用基于进程的线程时，最多需要启动的进程数量。
# use_multiprocessing：布尔值。当为True时，使用基于基于过程的线程。

testGene = testGenerator("data/membrane/test")   #批量导入图片并修改格式
results = model.predict_generator(testGene,30,verbose=1)  #测试数据进行预测
saveResult("data/membrane/test",results)   #批量保存结果