from __future__ import annotations
import pandas as pd
import yaml
from loguru import logger
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Sequence,
    Literal,
    Any,
    Tuple,
    Iterator,
    Mapping,
    Callable,
)


class SnakeFile(object):
    """
    SnakeMake Parser
    """

    def __init__(self):
        """
        SnakeMake Parser
        """
        self.header = None
        self.ruleLs = []
        self.all = None
        self.currentStep = 0
        self.resultDir = ""
        self.pipelineDir = ""
        self.main = ""

    def __str__(self):
        return f"Rules Name: {[x.name for x in self.ruleLs]}"

    def __repr__(self):
        return f"Rules Name: {[x.name for x in self.ruleLs]}"

    def addRule(self, snakeRule: SnakeRule):
        snakeRuleNameLs = [x.name for x in self.ruleLs]
        if snakeRule.name not in snakeRuleNameLs:
            self.ruleLs.append(snakeRule)
            self.currentStep += 1
            snakeRule.step = self.currentStep
        else:
            delIndex = snakeRuleNameLs.index(snakeRule.name)
            del self.ruleLs[delIndex]
            self.ruleLs.insert(delIndex, snakeRule)
            snakeRule.step = delIndex + 1

        logger.info(f"{snakeRule.name} step num: {self.currentStep}")

    def addHeader(self, snakeHeader: SnakeHeader):
        self.header = snakeHeader
        self.resultDir = "resultDir"
        self.pipelineDir = "pipelineDir"

    def addAll(self, snakeAll: SnakeAll):
        self.all = snakeAll

    def getMain(self, snakefilePath: Optional[str] = None):
        self.main = self.header.getMain() + "\n"
        for snakeRule in self.ruleLs:
            self.main = self.main + snakeRule.code + "\n"
        self.main = self.main + self.all.getMain() + "\n"
        for snakeRule in self.ruleLs:
            self.main = self.main + snakeRule.getMain() + "\n"
        print(self.main)
        if snakefilePath:
            with open(snakefilePath, "w") as fh:
                print(self.main, file=fh)


class SnakeHeader(object):
    def __init__(self, snakeFile: SnakeFile, configPath: str):

        """
        SnakeMake Parser
        Snakemake

        Parameters
        ----------
        snakeFile : SnakeFile
        configPath : str
            config should include two features:
                'pipelineDir', 'resultDir'
        """
        self.path = configPath
        self.yaml = yaml.load(open(self.path))
        self.snakeFile = snakeFile
        self.snakeFile.addHeader(self)
        if not (("pipelineDir" in self.yaml) & ("resultDir" in self.yaml)):
            logger.warning("pipelineDir or resultDir not in the yaml file")
        self.code = "import pandas as pd\n"
        self.code += f'#configfile: "{self.path}"\n'
        self.code += f"pipelineDir = config['pipelineDir']\n"
        self.code += f'resultDir = config["resultDir"].rstrip("/") + "/"\n'
        self.code += f'pipelineDir = config["pipelineDir"].rstrip("/") + "/"\n'

    def __str__(self):
        return self.code

    def __repr__(self):
        return self.__str__()

    def getConfig(self):
        return self.yaml

    def addCode(self, codeStr):
        self.code += codeStr

    def getMain(self):
        return self.code


class SnakeRule(object):
    """
    SnakeRule parser
    """

    def __init__(
        self,
        snakeFile: SnakeFile,
        name: str,
        threads: int,
        gpu: int = 0,
        outFile: str = "{sample}.finished",
        priority: int = 0,
    ):
        self.name = name
        self.step = None
        self.snakeFile = snakeFile
        self.snakeFile.addRule(self)  # step will be updated
        self.threads = threads
        self.gpu = gpu
        self.priority = f"{' ' * 4}priority:{priority}\n"

        self.code = ""
        self.metaDfName = ""
        self.needRuleLs = []
        self.input = " " * 4 + "input:\n"
        self.params = " " * 4 + "params:\n" + " " * 8 + f"gpu = {self.gpu},\n"
        self.outFile = outFile
        self.outputDir = self.snakeFile.resultDir + f" + 'step{self.step}_{self.name}/'"
        self.output = (
            " " * 4
            + "output:\n"
            + " " * 8
            + f"{self.name}Finished = {self.outputDir} + '{self.outFile}',\n"
        )
        self.threads = f"{' ' * 4}threads:{threads}\n"
        self.shell = ""
        self.main = f"rule {self.name}:\n"

    def __str__(self):
        return f"{self.code}{'-' * 16}\nIN RULE\n{'-' * 16}\n{self.getMain()}"

    def __repr__(self):
        return self.__str__()

    def addCode(self, code: str):
        self.code += code.strip("\n") + "\n"

    def addMetaDf(self, metaDfName: str, needAddRuleDirLs: Optional[Sequence[str]] = None):
        """
        add meta dataframe to snakerule. All meta information should store in this dataframe

        Parameters
        ----------
        metaDfName : str
            meta dataframe name, which is define in self.code
        needAddRuleDirLs : Optional[Sequence[str]]
            These columns will added by self.outputDir
        """
        self.metaDfName = metaDfName
        if needAddRuleDirLs:
            self.code += f"""
for column in {needAddRuleDirLs}:
    {metaDfName}[column] = {self.outputDir} + {metaDfName}[column]
""".lstrip(
                "\n"
            )
        # exec(self.code + f"\nprint(list({self.metaDfName}.columns))")

    def addMain(
        self,
        category: Literal["input", "params"],
        useColLs: Union[str, Sequence[str]],
        fromRule: Optional[SnakeRule] = None,
        gatherInfoDt: Mapping[str, str] = {},
        wildCardName: str = "sample",
    ):
        """
        add info to input\params\output based on dataframe

        Parameters
        ----------
        category : Literal["input", "params"]
            "input", "params", "output"
        useColLs : Union[str, Sequence[str]]
            These columns will be set as corresponding attributes
        fromRule : Optional[SnakeRule], optional
            Use which SnakeRule's metaDf, if None, will use self.
            by default None
        gatherInfoDt : Mapping[str, str], optional
            Whether need gather these infomation from different samples, key is col, value is the name of gathered groups.
            e.g. : {'inputFile': 'sampleLs'}
            by default {}.
        wildCardName : str, optional
            by default 'sample'
        """
        if isinstance(useColLs, str):
            useColLs = [useColLs]
        if not fromRule:
            fromRule = self
        else:
            self.input += " " * 8 + f"{fromRule.name}Finished = {fromRule.outputDir} + '{fromRule.outFile}',\n"

        metaInfoStr = ""
        for col in useColLs:
            if col in gatherInfoDt:
                colAttr = self.getAttrFromDfInSnakemake(
                    fromRule.metaDfName, col, wildCardName, gatherInfoDt[col]
                )
            else:
                colAttr = self.getAttrFromDfInSnakemake(
                    fromRule.metaDfName, col, wildCardName
                )
            metaInfoStr += " " * 8 + colAttr + ",\n"
        if category == "input":
            self.input += metaInfoStr
        elif category == "params":
            self.params += metaInfoStr
        else:
            assert False, f"Wrong Category: {category}"

    def setShell(self, shell: str):
        shell = shell.strip("\n") + "\n"
        self.shell = (
            " " * 4
            + "shell:\n"
            + " " * 8
            + '"""\n'
            + shell
            + f"touch {{output.{self.name}Finished}}\n"
            + " " * 8
            + '"""\n'
        )

    def getMain(self):
        return f"{self.main}{self.input}{self.output}{self.params}{self.threads}{self.priority}{self.shell}"

    @staticmethod
    def getAttrFromDfInSnakemake(
        dfName: str,
        col: str,
        wildCardName: str = "sample",
        rowNameLsName: Optional[str] = None,
    ) -> str:
        """
        if not providing <rowNameLsName>:

            getAttrFromDfInSnakemake('cellRangerMetaDf', 'input', 'sample')  ---->

                input = lambda wildcard: cellRangerMetaDf.at[wildcard.sample, 'input']

        if providing:

            getAttrFromDfInSnakemake('cellRangerMetaDf', 'input', 'sample', 'sampleLs')  ---->

                input = [(lambda x: cellRangerMetaDf.at[x, 'input'])(sample) for sample in sampleLs]

        Parameters
        ----------
        dfName : str
        col : str
        wildCardName : str, optional
            by default 'sample'
        rowNameLsName : Optional[str], optional
            by default None

        Returns
        -------
        str
        """
        if not rowNameLsName:
            return f"{col} = lambda wildcard: {dfName}.at[wildcard.{wildCardName}, '{col}']"
        else:
            return f"{col} = [(lambda x: {dfName}.at[x, '{col}'])({wildCardName}) for {wildCardName} in {rowNameLsName}]"


class SnakeAll(object):
    """
    snake rule all parser
    """

    def __init__(self, snakeFile: SnakeFile, *snakeRuleLs: Sequence[SnakeRule]):
        """
        snake rule all parser

        Parameters
        ----------
        snakeFile : SnakeFile
        *snakeRuleLs: rule terminals
        """
        import re

        self.snakeFile = snakeFile
        self.snakeFile.addAll(self)
        self.input = " " * 4 + "input:\n"

        for snakeRule in snakeRuleLs:
            wildcardLs = re.findall(r"{([\w\W]+)}", snakeRule.outFile)
            assert (
                len(wildcardLs) <= 1
            ), f"More than one wildcard found in {snakeRule.name}:{snakeRule.outFile}"
            if not wildcardLs:
                self.input += (
                    " " * 8 + f"{snakeRule.name}Finished = {snakeRule.outputDir} + '{snakeRule.outFile}',\n"
                )
            else:
                wildcard = wildcardLs[0]
                leftPartStr, rightPartStr = snakeRule.outFile.split(f"{{{wildcard}}}")
                self.input += (
                    " " * 8
                    + f'{snakeRule.name}Finished = [{snakeRule.outputDir} + "{leftPartStr}" + {wildcard} + "{rightPartStr}" for {wildcard} in {wildcard}Ls],\n'
                )


        self.main = "rule all:\n" + self.input

    def __str__(self):
        return self.getMain()

    def __repr__(self):
        return self.getMain()

    def getMain(self):
        return self.main
