import serial
import time

serialPort = "COM7"  # 串口
baudRate = 115200  # 波特率

ser = serial.Serial(serialPort, baudRate, timeout=0.5)

print("参数设置：串口=%s ，波特率=%d" % (serialPort, baudRate))

demo1="N"#将0转换为ASCII码方便发送
demo2="D"#同理

def senddata(data):
    ser.write(data.encode("gbk"))
    
while 1:
    time.sleep(1)
    senddata(demo1)
    time.sleep(1)
    senddata(demo2)
    # print("read:",ser.read())