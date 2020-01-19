
def prep_data():
    path = "omegle_dataset.txt"
    with open(path) as f:
        data = f.readlines()
    proccessed = []
    for line in data:
        proccessed.append(line.encode('ascii', 'ignore').decode('ascii'))
    with open("prepared_data.txt", "w") as f:
        f.write("")
    with open("prepared_data.txt", "a") as f:
        for pline in proccessed:
            if pline == "\n" or pline == "Stranger: \n":
                pass
            else:
                f.write(pline)
