import xml.dom.minidom

def tableau_vers_xml(tableau):
    document = xml.dom.minidom.Document()
    tab = document.createElement("tableau")
    document.appendChild(tab)
    for i in tableau:
        element = document.createElement("element")
        element.appendChild(document.createTextNode(str(i)))
        tab.appendChild(element)
    return document.toprettyxml()

def xml_vers_tableau(xmlchaine):
    tableau = []
    document = xml.dom.minidom.parseString(xmlchaine)
    for element in document.getElementsByTagName("element"):
        tableau.append(int(element.firstChild.data))
    return tableau


tableau = [1, 2, 3, 4, 5]
xmlchaine = tableau_vers_xml(tableau)
print(xmlchaine)
print(xml_vers_tableau(xmlchaine))

