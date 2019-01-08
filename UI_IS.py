from tkinter import *

root = Tk()

menu = Menu(root)
root.config(menu=menu)
root.title("Sign Language")
root.geometry("500x400")

"""
subMenu = Menu(menu)
menu.add_cascade(label="File", menu="subMenu")
subMenu.add_command(label="New Image")
subMenu.add_command(label="Add Image")
subMenu.add_separator()
subMenu.add_command(label="Exit")

editMenu = Menu(menu)
menu.add_cascade(label="edit", menu="editMenu")
editMenu.add_command(label="Redo")
"""

toolbar = Frame(root, bg="white")

insertButt = Button(toolbar, text="Insert Image")
insertButt.pack(side=LEFT, padx=10, pady=5)
printButt = Button(toolbar, text="Run")
printButt.pack(side=RIGHT, padx=10, pady=5)

toolbar.pack(side=BOTTOM, fill=X)

root.mainloop()
