"""
@Date: 2020-06-05 22:10:59
LastEditors: Liuzj
LastEditTime: 2020-09-14 11:20:49
@Description: 用于snakemake文件的生成
@Author: liuzj
FilePath: /liuzj/softwares/python_scripts/jpy_modules/jpy_tools/parseSnake.py
"""
import yaml
import logging
from loguru import logger
from typing import Optional
import re


def parseFilePathWithWildcards(filePath):
    wildCardInOutpusLs = re.findall(r"{{(\w+)}}", filePath)
    if wildCardInOutpusLs:
        outScripts = " ".join([f"for {x} in {x}Ls" for x in wildCardInOutpusLs])
        outScripts = f"[{filePath} {outScripts}]".replace(
            "{{", "{"
        ).replace("}}", "}")
    else:
        print(f"not found wildcard in filePath")
        0/0
    return outScripts

class SnakeMakeFile:
    def __init__(self):
        self.text = ""
        self.rules = {}
        self.pathPool = {}
        self.currentStep = 0

    def addHeader(self, snakeHeader):
        self.header = snakeHeader.text

    def addAll(self, snakeAll):
        self.all = snakeAll.text

    def addRule(self, snakeRule):
        self.rules[snakeRule.name] = snakeRule.text

    def generateContent(self, snakefilePath=False):
        self.text = f"{self.header}\n\n{self.all}\n\n"
        for x, y in self.rules.items():
            self.text += y

        if snakefilePath:
            print(self.text)
            with open(snakefilePath, "w") as fh:
                fh.write(self.text)
        else:
            print(self.text)
            return self.text


class SnakeHeader:
    """
    Snakemake config should include two features:
        'pipelineDir',
        'resultDir' 
    """

    def __init__(self, snakeFile: SnakeMakeFile, configPath: str):
        self.path = configPath
        self.yaml = yaml.load(open(self.path))
        if not (("pipelineDir" in self.yaml) & ("resultDir" in self.yaml)):
            logging.warn("pipelineDir or resultDir not in the yaml file")
        self.text = f'configfile: "{self.path}"\n'
        self.text += f"pipelineDir = config['pipelineDir']\n"
        self.snakeFile = snakeFile

    def addFeature(self, *features):
        """
        feature pipelineDir will be added auto
        """
        features = set(features)
        for singleFeature in features:
            self.text += f"{singleFeature} = config['{singleFeature}']\n"
        return self.text

    def addCode(self, codeStr):
        self.text += codeStr
    
    def addLsToPool(self, *sampleLs):
        for sample in sampleLs:
            self.snakeFile.pathPool[sample] = f'sample = \'{{{sample}}}\''

    def generateContent(self):
        print("config contents:\n")
        for key, value in self.yaml.items():
            print(f"{key:^20}:{value}")
        print('-----------------\n',self.text)
        self.snakeFile.addHeader(self)


class SnakeRule:
    def __init__(
        self,
        snakeFile: SnakeMakeFile,
        name: str,
        threads: int = 1,
        gpu: int = 0,
        step: Optional[int] = None,
    ):
        if not step:
            self.step = snakeFile.currentStep + 1
            snakeFile.currentStep = self.step
            self.step = str(self.step)
            logger.info(f"Current step: {self.step}")
        else:
            self.step = str(step)

        self.annotation = ""
        self.name = name
        self.threads = str(threads)
        self.gpu = f'"{gpu}"'

        self.input = ""
        self.output = ""
        self.params = f"    params:\n"
        self.params += f"        gpu = {self.gpu}\n"
        self.shell = ""

        self.snakeFile = snakeFile
        self.pathPool = self.snakeFile.pathPool
        self.resultDir = f"f\"{{config['resultDir']}}step{self.step}_{self.name}/\""
        # self.resultDir = f'step{self.step}ResultDir = f"{{config[\'resultDir\']}}step{self.step}_{self.name}/"'
        self.pathPool[f"step{self.step}ResultDir"] = self.resultDir

    def _parseSinglePath(self, path):
        pathSplit = path.split("/")
        if len(pathSplit) == 1:
            return path.split(".")[0]
        else:
            if pathSplit[-1] == "":
                path = pathSplit[-2]
                return path
            else:
                path = pathSplit[-1]
                return path.split(".")[0]

        # if len(path.split('/')) <= 2: # 如果
        #     if len(path.split('/')) == 1:
        #         pass
        #     else:
        #         if path.split('/')[1] == '':
        #             pass
        #         else:
        #             path = path.split('/')[-1]
        # else:
        #     path = path.split('/')[-1]

        # if len(path.split('.')) == 2:
        #     return path.split('.')[0]
        # else:
        #     if len(path.split('/')) == 1:
        #         return path.split('/')[0]
        #     else:
        #         return path.split('/')[-2]

    def _parseDict(self, IODict):
        returnList = []
        for pos, singleList in IODict.items():
            if (pos == "defaultDir") | (pos == "a"):
                for single in singleList:
                    fileName = self._parseSinglePath(single)
                    if fileName[0] == "{":
                        fileName = "_".join(fileName.split("_")[1:])
                    if fileName in self.pathPool:
                        returnList.append(self.pathPool[fileName])
                    else:
                        parsedPath = f"{fileName} = f\"{{config['resultDir']}}step{self.step}_{self.name}/{single}\""
                        self.pathPool[fileName] = parsedPath
                        returnList.append(parsedPath)

            elif (pos == "config") | (pos == "b"):
                for single in singleList:
                    fileName = single
                    if fileName in self.pathPool:
                        returnList.append(self.pathPool[fileName])
                    else:
                        parsedPath = f"{fileName} = config['{fileName}']"
                        self.pathPool[fileName] = parsedPath
                        returnList.append(parsedPath)

            elif (pos == "original") | (pos == "c"):
                assert isinstance(singleList, dict), "not dict"
                for singleName, singleContent in singleList.items():
                    self.pathPool[singleName] = f"{singleName} = {singleContent}"
                    returnList.append(f"{singleName} = {singleContent}")
            elif (pos == "wildcard") | (pos == "e"):
                for single in singleList:
                    singlePath = self.pathPool[single].split("=")[1].strip()
                    singlePath = parseFilePathWithWildcards(singlePath)
                    self.pathPool[f"all{single.capitalize()}"] = f"all{single.capitalize()} = {singlePath}"
                    returnList.append(f"{single} = {singlePath}")
            else:
                assert isinstance(singleList, dict), "not dict"
                for stepDir, singlePaths in singleList.items():
                    for singlePath in singlePaths:
                        store = True
                        if singlePath == '/':
                            singleName = f"{stepDir}ResultDir"
                            singlePath = ''
                            store = False
                        elif singlePath.endswith("/"):
                            singleName = singlePath.split("/")[-2]
                        else:
                            singleNameLs = singlePath.split("/")[-1].split(".")[:-1]
                            singleNameLs[1:] = [x.capitalize() for x in singleNameLs][
                                1:
                            ]
                            singleName = "".join(singleNameLs)

                        stepDirPath = self.pathPool[f"{stepDir}ResultDir"][:-1]
                        singleContent = stepDirPath + singlePath + '"'
                        if store:
                            self.pathPool[singleName] = f"{singleName} = {singleContent}"
                        returnList.append(f"{singleName} = {singleContent}")
        return returnList

    def setAnnotation(self, anno):
        self.annotation = "#" + anno + "\n"

    def _parseIO(self, IODict):
        returnList = []
        returnList.extend(self._parseDict(IODict))
        returnStr = ",\n        ".join(returnList)
        returnStr = "        " + returnStr
        return returnStr

    def setInput(self, **inputs):
        """
        a | defaultDir :
            ['illuminaParsed.index', 'ref/', {sample}_rawMapped.bam]
        b | config :
            ['window']
        c | original :
            dict(gap = -12)
        d:
            dict(step2 = ['polyACluster/polya_cluster.filtered.bed', 'testDir/']) resulted: polya_clusterFiltered, testDir
        e:
            use wildcard parser. ['window']
        """
        self.input = self._parseIO(inputs)
        self.input = f"    input:\n" + self.input
        return self.input

    def setOutput(self, **outputs):
        """
        a | defaultDir :
            ['illuminaParsed.index', 'ref/', {sample}_rawMapped.bam]
        b | config :
            ['window']
        c | original :
            dict(gap = -12)
        d:
            dict(step2 = ['polyACluster/polya_cluster.filtered.bed', 'testDir/']) resulted: polya_clusterFiltered, testDir
        e:
            use wildcard parser. ['window']
        """
        self.output = self._parseIO(outputs)
        self.output = f"    output:\n" + self.output
        return self.output

    def setParams(self, **params):
        """
        a | defaultDir :
            ['illuminaParsed.index', 'ref/', {sample}_rawMapped.bam]
        b | config :
            ['window']
        c | original :
            dict(gap = -12)
        d:
            dict(step2 = ['polyACluster/polya_cluster.filtered.bed', 'testDir/']) resulted: polya_clusterFiltered, testDir
        e:
            use wildcard parser. ['window']
        """
        self.params = self._parseIO(params)
        self.params = f"    params:\n" + self.params
        self.params += f",\n        gpu = {self.gpu}\n"
        return self.params

    def setShell(self, commands):
        self.shell = "    shell:\n"
        self.shell += '        """\n'
        # self.shell += f'jpy_qsub.py --sm -t {{threads}} -n {self.node} -N {self.name} --inline \'\\\n'
        self.shell += f"cd {{pipelineDir}}\n"
        self.shell += commands.strip()
        # self.shell += '\''
        self.shell += '\n        """'
        return self.shell

    def generateContent(self):
        text = self.annotation
        text += f"rule {self.name}:\n"
        text += f"{self.input}\n"
        text += f"{self.output}\n"
        text += f"{self.params}\n"
        text += f"    threads:{self.threads}\n"
        text += self.shell
        text += "\n\n"
        self.text = text
        print(self.text)
        self.snakeFile.addRule(self)


class SnakeAll:
    def __init__(self, snakeFile):
        self.snakeFile = snakeFile
        self.pathPool = self.snakeFile.pathPool
        self.outputs = []

    def addOutput(self, *outputs):
        self.outputs = outputs

    def generateContent(self):
        text = "rule all:\n"
        text += "    input:\n"
        outputTemp = []

        for singleOutput in self.outputs:
            singleOutput = self.pathPool[singleOutput].split("=")[1].strip()
            wildCardInOutpusLs = re.findall(r"{{(\w+)}}", singleOutput)
            if wildCardInOutpusLs:
                outScripts = " ".join([f"for {x} in {x}Ls" for x in wildCardInOutpusLs])
                outScripts = f"[{singleOutput} {outScripts}]".replace(
                    "{{", "{"
                ).replace("}}", "}")
            else:
                outScripts = singleOutput
            outputTemp.append(outScripts)

        outputs = ",\n        ".join(outputTemp)
        text += "        "
        text += outputs
        self.text = text
        print(self.text)
        self.snakeFile.addAll(self)

    # def generateContent(self, **outputs):
    #     """
    #     outputs:
    #     {'illuminaParsed': 0, 'mappedDir' : 1}
    #     if mark is 1, will ues expand grammer.
    #     and sampleList should output by SnakeConfig
    #     """
    #     text = 'rule all:\n'
    #     text += '    input:\n'
    #     outputTemp = []
    #     for name, mark in outputs.items():
    #         if mark == 0:
    #             singleText = self.pathPool[name]
    #             outputTemp.append(singleText)
    #         else:
    #             singleText = self.pathPool[name].split('=')[1].strip()
    #             singleText = f'[{singleText} for sample in config[\'sampleList\']]'.replace('{{sample}}', '{sample}')
    #             outputTemp.append(singleText)
    #     outputs = ',\n        '.join(outputTemp)
    #     text += '        '
    #     text += outputs
    #     self.text = text
    #     self.snakeFile.addAll(self)
