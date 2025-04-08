#  build the class
import numpy
import scipy.special
import csv
import cv2

class neuralnetwork:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learnrate):
        #  set numbers of nodes
        self.inode = inputnodes;
        self.hnode = hiddennodes;
        self.onode = outputnodes;
        self.lr = learnrate;

        # weights：i&h,h&o between input&hidden hidden&output use random number to initialize
        # minus 0.5 to make sure weights can be negative

        self.wih = (numpy.random.rand(self.hnode,self.inode)-0.5)
        self.who = (numpy.random.rand(self.onode,self.hnode)-0.5)

        # activation function is sigmond function(expit())
        # use lambda to build a function instead of  def: / the function created by lambda do not have name; here
        # we give a name activation-function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self,input_lists,target_lists):

        #convertion: transform array to matrix
        inputs = numpy.array(input_lists,ndmin=2).T

        target = numpy.array(target_lists,ndmin=2).T

        # similar to query: used to check the outputs

        hidden_inputs = numpy.dot(self.wih,inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        #calculate errors

        final_errors = target-final_outputs

        #backward in weight in ratio

        # 原本是hidden dot weight = final errors 两边同时乘转置
        hidden_errors = numpy.dot(self.who.T,final_errors)

        #update the weight

        self.who+=self.lr*numpy.dot((final_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))

        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))
        pass


    def query(self,inputs_lists):
        # imputs_lists是一个一维表 通过下面一行可以转换成一个二维的数组： numpy.array 将array转换成矩阵
        inputs = numpy.array(inputs_lists, ndmin=2).T

        # 下面的四行用于构建输入和输出
        #hidden inputs 是输入层的输出点成各个权重 构成隐藏层的各个输入

        hidden_inputs = numpy.dot(self.wih,inputs)

        # 隐藏层的输出就是输入运用到function之中
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        return final_outputs



    # save_weights is used to save the data after training
    def save_weights(self):
        with open('weight_data.csv',mode='w',newline='') as csv_file:
            writer = csv.writer(csv_file)
            for row in self.wih:
                writer.writerow(row)
                # write in -
            writer.writerow(['---'])
            # write in who
            for row in self.who:
                writer.writerow(row)


    # load the saved weights into new ANN
    def load_data(self):
        input_hidden_weights = []
        hidden_output_weights = []
        is_input_hidden = True
        with open('weight_data.csv', mode='r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if row[0] == '---':
                    is_input_hidden = False
                    continue
                if is_input_hidden:
                    input_hidden_weights.append([float(val) for val in row])
                else:
                    hidden_output_weights.append([float(val) for val in row])
        self.wih = numpy.array(input_hidden_weights)
        self.who = numpy.array(hidden_output_weights)


    # show the capture picture and the ANN checking output
    def show_output(self, data, frame):
        # 将识别结果转换为字符串
        output_text = str(data)
        # 在图像上添加文本
        cv2.putText(frame, f"Recognized: {output_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame
