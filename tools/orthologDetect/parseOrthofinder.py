import glob
from itertools import product
from tqdm import tqdm
import pandas as pd
import yaml
import sh
import click


@click.command()
@click.option("-i", "dir_result")
@click.option("-o", "dir_out")
@click.option("-p", "path_config")
def main(dir_result, dir_out, path_config):
    """
    parse orthofinder results

    \b
    Parameters
    ----------
    -i, dir_result :
        orthofinder `Orthologues` result dir
    -o, dir_out :
        output dir
    -p, path_config :
        path_config
    """
    sh.mkdir(dir_out, p=True)
    with open(path_config) as fh:
        dt_lambda = yaml.safe_load(fh)
    dt_lambda = {x: eval(f"lambda {y}") for x, y in dt_lambda.items()}

    ls_name = [
        x.split("/")[-1].replace("Orthologues_", "", 1)
        for x in glob.glob(f"{dir_result}/*")
    ]

    assert sorted(ls_name) == sorted(
        list(dt_lambda.keys())
    ), f"There is a problem with the keys of lambda config: \n input: {ls_name}\n config: {list(dt_lambda.keys())}"

    dt_name2path = {}
    for x, y in product(range(len(ls_name)), range(len(ls_name))):
        if x < y:
            x_name = ls_name[x]
            y_name = ls_name[y]
            dt_name2path[
                f"{x_name}__v__{y_name}"
            ] = f"{dir_result}/Orthologues_{x_name}/{x_name}__v__{y_name}.tsv"

    for name, path in tqdm(dt_name2path.items()):
        df = pd.read_table(path)
        df.drop(columns="Orthogroup", inplace=True)

        for columnName in df.columns:
            df[columnName] = df[columnName].str.split(", ")
            df = df.explode(columnName)
        df = df.reset_index(drop=True)

        for columnName in df.columns:
            df[columnName] = df[columnName].map(dt_lambda[columnName])

        df.drop_duplicates(inplace=True, keep=False)

        for columnName in df.columns:
            df.drop_duplicates([columnName], inplace=True)

        df.to_csv(f"{dir_out}/{name}.1v1.tsv", index=None, sep="\t")


main()