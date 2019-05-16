# encoding: utf-8


class Class_info:
	class_names = ['background','ASUS_Z97-AR','MSI_Z87-G45_Gaming','Gateway_TBGM-01', 'PCI-express_x16','cpu_socket','6pinslot','8pinslotgpu','8pinslotcpu','24pinslot','sata_slot','cpu_fanslot','fan_power_slot','ssdhddpowerslot','ramslot','ssd','hdd','ddr3','ddr4','ssdhdd_powercable','gpu','cpu','cpu_fan','fan_powercable','psu','6pincable','8pincablegpu','8pincablecpu','24pincable','sata_cable']
	num_classes = len(class_names)
	classes_originID = dict((i,j) for (i,j) in zip(class_names, [i for i in range(num_classes)]) if i != 'background')

class_info = Class_info()
#print(class_info.num_classes)
