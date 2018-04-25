import win32clipboard as wc
import win32con
 
def getClipboard():
	wc.OpenClipboard()
	bintxt = wc.GetClipboardData(win32con.CF_TEXT)
	wc.CloseClipboard()
	return bintxt 

def stripSpace(bintxt):
	return bintxt.decode('utf-8',errors='ignore').replace('\r\n', ' ').encode('utf-8')

def writeClipboard(txt):
	wc.OpenClipboard()
	wc.EmptyClipboard()
	wc.SetClipboardData(win32con.CF_TEXT, txt)
	wc.CloseClipboard()

if __name__ == "__main__":
	binTxt = getClipboard()
	txt = stripSpace(binTxt)
	writeClipboard(txt)
