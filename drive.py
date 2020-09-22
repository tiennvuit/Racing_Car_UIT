import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import cv2
from PIL import Image
from flask import Flask
from io import BytesIO

#------------- Add library ------------#

from keras.models import load_model
import argparse
import utils
import os
#--------------------------------------#

#Global variable
MAX_SPEED = 25
MAX_ANGLE = 25

# Tốc độ thời điểm ban đầu
speed_limit = MAX_SPEED
MIN_SPEED = 15

#init our model and image array as empty
model = None
prev_image_array = None
previous_angle = 0
previous_speed = 0

# My team define
previous_image = None
save_image_path = "data/train/train_map1"
grouth_truth = "data/train/grouth_truth/grouth_truth_map1.txt"
index = 0

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    
    global index
    global previous_image
    global previous_angle
    global previous_speed

    if data:

        steering_angle = 0  #Góc lái hiện tại của xe
        speed = 0           #Vận tốc hiện tại của xe
        image = 0           #Ảnh gốc

        steering_angle = float(data["steering_angle"])
        speed = float(data["speed"])

        #Original Image
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # store label
        if index % 5 == 0:
            with open(grouth_truth, 'a+') as f:
                f.write("{}, {}, {}, {}, {}\n".format("{:08d}".format(index), 
                    previous_angle, previous_speed, steering_angle, speed))

        previous_angle = steering_angle
        previous_speed = speed

        print("{} --> Previous: {}-{}, Current: {}-{}\n".format("{:08d}".format(index), 
                    previous_angle, previous_speed, steering_angle, speed))
        """
        - Chương trình đưa cho bạn 3 giá trị đầu vào:
            * steering_angle: góc lái hiện tại của xe
            * speed: Tốc độ hiện tại của xe
            * image: hình ảnh trả về từ xe
            * depth_image: ảnh chiều sâu được xe trả về (xét takeDepth = True, ảnh depth sẽ được trả về sau khi 'send_control' được gửi đi)

        - Bạn phải dựa vào 3 giá trị đầu vào này để tính toán và gửi lại góc lái và tốc độ xe cho phần mềm mô phỏng:
            * Lệnh điều khiển: send_control(sendBack_angle, sendBack_Speed)
            Trong đó:
                + sendBack_angle (góc điều khiển): [-25, 25]  NOTE: ( âm là góc trái, dương là góc phải)
                + sendBack_Speed (tốc độ điều khiển): [-150, 150] NOTE: (âm là lùi, dương là tiến)
        """
        sendBack_angle = 0
        sendBack_Speed = 0
        try:
            #------------------------------------------  Work space  ----------------------------------------------#

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = utils.preprocess(image)

            # store data image
            if index % 5 == 0:
                try:
                    name = os.path.join(save_image_path, "{:08d}.png".format(index))
                    cv2.imwrite(name, previous_image)
                except:
                    pass

            previous_image = image.copy()

            index = index + 1

            cv2.imshow("Origin frame", image)
            #cv2.imshow("Gray frame", img_gray)
            cv2.imshow("Utils image", image)
            cv2.waitKey(1)


            image = np.array([image])


            # try:
            #     steering_angle = float(model.predict(image, batch_size=1))
            # except:
            #     pass

            # # Tốc độ ta để trong khoảng từ 10 đến 25
            # global speed_limit
            # if speed > speed_limit:
            #     speed_limit = MIN_SPEED  # giảm tốc độ
            # else:
            #     speed_limit = MAX_SPEED
            # throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2


            # sendBack_angle = steering_angle*MAX_ANGLE
            # sendBack_Speed = throttle*MAX_SPEED

            

            #------------------------------------------------------------------------------------------------------#
            print('{} : {}'.format(sendBack_angle, sendBack_Speed))
            send_control(sendBack_angle, sendBack_Speed)


        except Exception as e:
            cv2.destroyAllWindows()
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)


if __name__ == '__main__':

    #-----------------------------------  Setup  ------------------------------------------#

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )

    # parser.add_argument(
    #     'data_output', type=str, default='data/train/', help="The path store image data"
    # )
    # parser.add_argument(
    #     'grouth_truth', type=str, default='data/grouth_truth.txt', help="The grouth truth file store labels"
    # )
    # args = parser.parse_args()

    # Load model mà ta đã train được từ bước trước
    try:
        model = load_model(args.model)
    except:
        print("Not found trained model")
    #--------------------------------------------------------------------------------------#
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    cv2.destroyAllWindows()
