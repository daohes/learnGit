from numpy import *
from tkinter import *
import regTrees
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
	reDraw.f.clf()
	reDraw.a = reDraw.f.add_subplot(111)
	if chkBtnVar.get():
		if tolN < 2: tolN = 2
		myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf, regTrees.modelErr, (tolS, tolN))
		yHat = regTrees.createForeCast(myTree, reDraw.testDat, regTrees.modelTreeEval)
	else:
		myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
		yHat = regTrees.createForeCast(myTree, reDraw.testDat)
	reDraw.a.scatter(reDraw.rawDat[:,0].A, reDraw.rawDat[:,1].A, s=20, c='g', alpha=.4)
	reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0, c='r')
	reDraw.canvas.show()

def getInputs():
	try:
		tolN = int(tolNentry.get())
	except:
		tolN = 10
		print("tolN应为整数")
		tolNentry.delete(0, END)
		tolNentry.insert(0, '10')
	try:
		tolS = float(tolSentry.get())
	except:
		tolS = 1.0
		print("tolS应为小数")
		tolNentry.delete(0, END)
		tolNentry.insert(0, '1.0')
	return tolN, tolS	

def drawNewTree():
	tolN, tolS = getInputs()
	reDraw(tolS, tolN)

root = Tk()
Label(root, text='Plot Place Holder').grid(row=0, columnspan=3)
Label(root, text='tolN').grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0,'10')
Label(root, text='tolS').grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0,'1.0')
Button(root, text='重画', command=drawNewTree).grid(row=1, column=2, rowspan=3)
#Button(root, text='Quit', fg='black', command=root.quit).grid(row=3, column=2)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text='模型树模型', variable = chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.f = Figure(figsize=(5,4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]), max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0, 10)
root.mainloop()

