import xmltodict
import json
import os
import fire


class XML2JSON:
    def fuck(self, fxml, fjson=None):
        if fjson is None:
            fjson = os.path.splitext(fxml)[0] + ".json"
        xdict = xmltodict.parse(open(fxml))
        json.dump(xdict, open(fjson, "w"), indent=4)
        print "saved to", fjson


if __name__ == "__main__":
    fire.Fire(XML2JSON)