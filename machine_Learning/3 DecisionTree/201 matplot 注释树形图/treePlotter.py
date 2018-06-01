import matplotlib.pyplot as plt
decisionNode = dict(boxstyle="sawtooth", fc=".8")
leafNode = dict(boxstyle="round4", fc=".8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction",\
                            xytext=centerPt, textcoords="axes fraction",\
                            va="center", ha="center", bbox=nodeType, \
                            arrowprops = arrow_args)
def createPlot():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    createPlot.axl = plt.subplot(111, frameon = False)
    plotNode("决策节点", (.5,.1),(.1,.5), decisionNode)
    plotNode("叶节点",(.8,.1),(.3,.8), leafNode)
    plt.show()

if __name__ == "__main__":
    createPlot()
