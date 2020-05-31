"""Task definitions for `inv(oke)`.
"""
import sys
import time

from invoke import task

# PYTHON_INTERPRETER = os.path.splitext(os.path.basename(sys.executable))[0]
PYTHON_INTERPRETER = sys.executable
WORKFLOW = [
    'data_download',
    'data_preprocess',
    'data_sample',
    'data_postprocess',
    'model_train',
    'model_evaluate',
    'visualize',
    'report',
]
WORKFLOW_TASK_HELP = {
    'config_name': 'Configuration name, i.e. file name of the '
    'configuration without the `json` extension.',
    'predecessors': 'Indicates if all the predecessor tasks in the workflow '
    'should also be run.',
    'successors': 'Indicates if all the successor tasks in the workflow '
    'should also be run.'
}


def run_py(context, string):
    """Pass given string to python interpreter within given `invoke` context.

    :param context: Invoke context.
    :type context: Context
    :param string: String to be passed to python interpreter.
    :type string: str
    :return: True if execution succeeded, False otherwise.
    :rtype: bool
    """
    try:
        context.run(f'{PYTHON_INTERPRETER} {string}')
        return True
    except Exception:
        return False


def run_workflow_tasks(
    context,
    tasks,
    config_name
):
    """Run given list of workflow tasks with provided configuration within given
    `invoke` context.

    :param context: Invoke context.
    :type context: Context
    :param tasks: List of workflow tasks to run.
    :type tasks: list[str]
    :param config_name: 'Configuration name, i.e. file name of the
        configuration without the `json` extension.',
    :type config_name: str
    """
    timestamp = time.time()

    for t in tasks:
        cmd = f'src/tasks/{t}.py {config_name} --timestamp {timestamp}'
        print(f'Running: {cmd}')

        success = run_py(context, cmd)

        if not success:
            print('Terminated!')
            break


def get_task_workflow(task, predecessors=False, successors=False):
    """Get a sequence of tasks from the workflow.

    :param task: Chosen task.
    :type task: str
    :param predecessors: Indicates if all the predecessor tasks in the workflow
        should also be run. Defaults to False.
    :type predecessors: bool, optional
    :param successors: Indicates if all the successor tasks in the workflow
        should also be run. Defaults to False.
    :type successors: bool, optional
    :return: Sequence of workflow tasks.
    :rtype: list[str]
    """
    task_index = WORKFLOW.index(task)
    workflow = [task]

    if predecessors:
        workflow = WORKFLOW[:task_index] + workflow

    if successors:
        workflow = workflow + WORKFLOW[(task_index + 1):]

    return workflow


@task(help=WORKFLOW_TASK_HELP)
def download(c, config_name, predecessors=False, successors=False):
    """Download data for given configuration.
    """
    run_workflow_tasks(
        c,
        get_task_workflow('data_download', predecessors, successors),
        config_name
    )


@task(help=WORKFLOW_TASK_HELP)
def preprocess(c, config_name, predecessors=False, successors=False):
    """Preprocess data for given configuration.
    """
    run_workflow_tasks(
        c,
        get_task_workflow('data_preprocess', predecessors, successors),
        config_name
    )


@task(help=WORKFLOW_TASK_HELP)
def sample(c, config_name, predecessors=False, successors=False):
    """Sample data for given configuration.
    """
    run_workflow_tasks(
        c,
        get_task_workflow('data_sample', predecessors, successors),
        config_name
    )


@task(help=WORKFLOW_TASK_HELP)
def postprocess(c, config_name, predecessors=False, successors=False):
    """Postprocess data for given configuration.
    """
    run_workflow_tasks(
        c,
        get_task_workflow('data_postprocess', predecessors, successors),
        config_name
    )


@task(help=WORKFLOW_TASK_HELP)
def train(c, config_name, predecessors=False, successors=False):
    """Train model for given configuration.
    """
    run_workflow_tasks(
        c,
        get_task_workflow('model_train', predecessors, successors),
        config_name
    )


@task(help=WORKFLOW_TASK_HELP)
def evaluate(c, config_name, predecessors=False, successors=False):
    """Evaluate model for given configuration.
    """
    run_workflow_tasks(
        c,
        get_task_workflow('model_evaluate', predecessors, successors),
        config_name
    )


@task(help=WORKFLOW_TASK_HELP)
def visualize(c, config_name, predecessors=False, successors=False):
    """Visualize data and results for given configuration.
    """
    run_workflow_tasks(
        c,
        get_task_workflow('visualize', predecessors, successors),
        config_name
    )


@task(help=WORKFLOW_TASK_HELP)
def report(c, config_name, predecessors=False, successors=False):
    """Generate reports for given configuration.
    """
    run_workflow_tasks(
        c,
        get_task_workflow('report', predecessors, successors),
        config_name
    )


@task()
def run(c, config_name):
    """Run the entire workflow:
        - download
        - preprocess
        - sample
        - postprocess
        - train
        - evaluate
        - visualize
        - report
    """
    run_workflow_tasks(c, WORKFLOW, config_name)


@task()
def info(c, config_name):
    """Print configuration info.
    """
    print(PYTHON_INTERPRETER)
    run_py(c, f'src/tasks/info.py {config_name}')


@task
def lint(c):
    """Lint python source code using flake8.
    """
    c.run('flake8 src')


@task
def install(c):
    """Install this project as a pip package.
    """
    c.run('pip install --editable .')
