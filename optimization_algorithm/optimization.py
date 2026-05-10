import random


class Process_Information:
    def __init__(self, information_num):
        self.information_num = information_num
        self.C_ELE = 0.82
        self.SI_ELE = 0.21
        self.MN_ELE = 0.52
        self.P_ELE = 0.011
        self.S_ELE = 0.006
        self.CR_ELE = 0.012
        self.NI_ELE = 0.007
        self.CU_ELE = 0.01
        self.ORT = 883
        self.SPEED1 = 0.82
        self.SPEED2 = 0.88
        self.SPEED3 = 0.97
        self.SPEED4 = 1.08
        self.SPEED5 = 1.10
        self.SPEED6 = 1.15
        self.SPEED7 = 1.16
        self.SPEED8 = 1.16
        self.SPEED9 = 1.16
        self.SPEED10 = 1.15
        self.FAN1 = 0.0
        self.FAN2 = 50.0
        self.FAN3 = 0.0
        self.FAN4 = 0.0
        self.FAN5 = 0.0
        self.FAN6 = 0.0

def get_many_information(n):
    if n <= 0:
        return []

    processinformation = []
    for i in range(1, n + 1):
        info = Process_Information(i)

        # ORT: 吐丝温度范围 [850, 900]
        info.ORT = random.randint(850, 900)

        # SPEED1~SPEED10: 辊道速度范围 [0.5, 1.5]
        for j in range(1, 11):
            setattr(info, f"SPEED{j}", random.uniform(0.5, 1.5))

        # FAN1~FAN6: 风机开度范围 [0, 100]
        for j in range(1, 7):
            setattr(info, f"FAN{j}", random.uniform(0.0, 100.0))

        processinformation.append(info)

    return processinformation

def GA():
    pass 

def PSO():
    pass


if __name__ == "__main__":
    processinformation = get_many_information(10)
    

