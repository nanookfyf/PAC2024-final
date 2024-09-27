#define _GNU_SOURCE
#include <unistd.h>
#include <sched.h>
#include <stdio.h>
#include <time.h>
#include <sys/wait.h>
#include <errno.h>



void sleep_us(unsigned usec) {
	_Thread_local static struct timespec t;
	t.tv_nsec = (usec % 1000000) * 1000;
	t.tv_sec = usec / 1000000;
	while(t.tv_nsec > 0 && nanosleep(&t, &t) == EINTR);
}

void clearnode(pid_t pid) {
	cpu_set_t *cpus = CPU_ALLOC(576);
	CPU_ZERO(cpus);
	for (int i = 0; i < 576; i++) {
		CPU_SET(i, cpus);
	}
	sched_setaffinity(pid, 576, cpus);
	CPU_FREE(cpus);
}

void setnode(pid_t pid, int node) {
	cpu_set_t *cpus = CPU_ALLOC(576);
	CPU_ZERO(cpus);
	for (int i = 0; i < 144; i++) {
		CPU_SET(i + node * 144, cpus);
	}
	sched_setaffinity(pid, 576, cpus);
	CPU_FREE(cpus);
}

int main() {
	char * argv[3];
	argv[0] = "./polynomial_stencil";
	argv[1] = "test2.conf";
	argv[2] = NULL;

	pid_t pid = fork();
	if(pid == 0) {
		execve(argv[0], argv, NULL);
	}
	
	/* init */
	sleep_us(7000);

	/* array fa */
	for (int i = 0; i < 4; i++) {
		//printtime();
		setnode(pid, i);
		sleep_us(78100);
	}
	
	/* array fb */
	//printtime();
	setnode(pid, 0);
	sleep_us(313000);

	/* array f random filling */
	for (int i = 0; i < 4; i++) {
	//	printtime();
		setnode(pid, i);
		sleep_us(4775000);
	}
	//printtime();

	/* maybe unnecessary */
	clearnode(pid);

	/* wait for and reap the child */
	waitpid(pid, NULL, 0);
}
