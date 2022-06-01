from numpy import power


class Hero:
    jumlah = 0
    def __init__(self, name, power, armor, health):
        #instance Variabel
        self.name = name
        self.power = power
        self.armor = armor
        self.healt = health
        Hero.jumlah += 1
    
    def serang(self, lawan):
        print(self.name + ' menyerang ' + lawan.name)
        lawan.diserang(self)
    def diserang(self, lawan, attPower_lawan):
        print(self.name + ' diserang ' + lawan.name)
        attack_diterima = attPower_lawan/self.armor
        print('serangan terasa : ' + str(attack_diterima))
        self.healt -= attack_diterima

sniper = Hero('sniper', 100, 15 , 5)
rikimaru = Hero('rikimaru', 100, 5, 10)

sniper.serang(rikimaru)