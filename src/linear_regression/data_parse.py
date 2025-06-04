


file = []
with open("./Housing.csv", "r") as f:
    file = f.readlines()


parsed_data_in = []
parsed_data_out = []

print(len(file))
for line in file[1:]:
    #price,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus
    features = line.split(",")
    tset = ""
    for f in features[1:]:
        f = f.strip()
        if f == "yes":
            tset += "1.0f,"
        elif f == "no":
            tset += "0.0f, "
        elif f == "furnished":
            tset += "1.0f, "
        elif f == "semi-furnished":
            tset += "0.5f, "
        elif f == "unfurnished":
            tset += "0.0f, "
        else:
            tset += f"{float(f)/1000}, "
    parsed_data_in.append("{" + f"{tset}" + "},") # features
    parsed_data_out.append(f"{float(features[0])/1000}") # house price
    price = features[0]
    area = features[1]
    bedrooms = features[2]
    bathrooms = features[3]
    stories = features[4]
    mainroad = features[5]
    guestroom = features[6]
    basement = features[7]
    hwheating = features[8]
    ac = features[9]
    parking = features[10]
    prefarea = features[11]
    furnishings = features[12]


with open("./data.h", "w") as f:
    f.write("#include <vector>\n")
    f.write("std::vector<std::vector<double>> data_in = {\n")
    for feat in parsed_data_in:
        f.write(feat + "\n")
    f.write("};\n\n")

    f.write("std::vector<double> data_out = {")
    for actual in parsed_data_out:
        f.write(actual + ",\n")
    f.write("};")