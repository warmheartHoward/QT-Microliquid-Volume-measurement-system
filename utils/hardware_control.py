import serial
import serial.tools.list_ports
import time


class Connecting_Serial:
    def __init__(self):
        self._b = bytearray()

    # 用于将指令集转换为2位16进制码，并保存在self._b中，打包
    def pac(self, addr: int, idd: int, paras: list[str]):
        self._b.append((addr & 0xff00) >> 8)
        self._b.append((addr & 0x00ff))
        self._b.append((idd & 0xff00) >> 8)
        self._b.append((idd & 0x00ff))

        self.addr = addr
        self.idd = idd
        self.paras = paras
        assert paras.__len__() == 8
        for item in paras:
            self._b.append(item & 0xff)

    def getbytearr(self):
        return self._b
    # 解包
    def unpac(self, raw: bytearray):
        assert raw.__len__() == 12
        self.addr = (raw[0] << 8) | raw[1]
        self.idd = (raw[2] << 8) | raw[3]
        for item in raw[4:]:
            self.paras.append(item)


# 用于控制泵
class Pump_MultiValve_Control(Connecting_Serial):
    def __init__(self, port_num:str="COM7"):
        self.rotating_speed = int(str(50), 16)
        self.direction = "Positive"
        self.Open_code = [1, 0x2003, [0x01, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, self.rotating_speed]]  # 控制泵转动的主板命令
        self.Close_code = [1, 0x2003, [0x00, 0x01, 0xff, 0xff, 0xff, 0xff, 0x00, self.rotating_speed]]  # 控制泵关闭的主板命令
        self.Valve_code = [1, 0x2005, [0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]]  # 控制阀切换的主板命令
        self.position = 0

        # 打开串口
        self.ser = serial.Serial(port=port_num, baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE, timeout=0.5)

    # 打开蠕动泵，设置蠕动泵方向, 转速
    def open_pump(self, direction: str, rotating_speed: int):
        self.direction = direction
        self.rotating_speed = int(str(rotating_speed), 16)
        if self.ser.isOpen():  # 判断串口是否成功打开
            print("串口连接成功")
            if self.direction == "Positive":
                self.Open_code = [1, 0x2003, [0x01, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, self.rotating_speed]]
            elif self.direction == "Negative":
                self.Open_code = [1, 0x2003, [0x01, 0x01, 0xff, 0xff, 0xff, 0xff, 0x00, self.rotating_speed]]
            else:
                raise Exception("The type of direction input is wrong")
            Connecting_Serial.__init__(self)
            Connecting_Serial.pac(self, self.Open_code[0], self.Open_code[1], self.Open_code[2])
            input_len = self.ser.write(Connecting_Serial.getbytearr(self))
            print("打开蠕动泵，向串口发出{}个字节。".format(input_len))

        else:
            print("打开串口失败。")
    # 关闭蠕动泵
    def close_pump(self):
        if self.ser.isOpen():  # 判断串口是否成功打开
            print("串口连接成功")
            Connecting_Serial.__init__(self)
            Connecting_Serial.pac(self, self.Close_code[0], self.Close_code[1], self.Close_code[2])
            input_len = self.ser.write(Connecting_Serial.getbytearr(self))
            print("关闭蠕动泵。向串口发送{}个字节".format(input_len))
        else:
            print("打开串口失败。")

    def adjust_valve_postion(self, position: int):  # position 从01-10选择一个端口
        self.position = int(str(position), 16)
        if self.ser.isOpen():  # 判断串口是否成功打开
            print("串口连接成功")
            self.Valve_code = [1, 0x2005, [self.position, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]]  # 控制阀切换的主板命令
        Connecting_Serial.__init__(self)
        Connecting_Serial.pac(self, self.Valve_code[0], self.Valve_code[1], self.Valve_code[2])
        input_len = self.ser.write(Connecting_Serial.getbytearr(self))
        print(f"调整多为阀端口为{position},向串口发送{input_len}个字节")

    def close_serial(self):
        self.ser.close()
















