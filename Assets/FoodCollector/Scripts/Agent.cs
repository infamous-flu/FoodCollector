using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AustinHarris.JsonRpc;

public class MyVector3
{
    public float x;
    public float y;
    public float z;
    public MyVector3(Vector3 v)
    {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
    }
    public Vector3 ToVector3()
    {
        return new Vector3(x, y, z);
    }
}

public class RLResult
{
    public float reward;
    public bool finished;
    public MyVector3 observation;
    public RLResult(float reward, bool finished, MyVector3 observation)
    {
        this.reward = reward;
        this.finished = finished;
        this.observation = observation;
    }
}

public class Agent : MonoBehaviour
{
    class Rpc : JsonRpcService
    {
        Agent agent;
        public Rpc(Agent agent)
        {
            this.agent = agent;
        }

        [JsonRpcMethod]
        void say(string message)
        {
            Debug.Log($"you sent {message}");
        }

        [JsonRpcMethod]
        MyVector3 getPosition()
        {
            return new MyVector3(agent.transform.position);
        }

        [JsonRpcMethod]
        RLResult step(string action) 
        {
            return agent.Step(action);
        }

        [JsonRpcMethod]
        MyVector3 reset()
        {
            return agent.Reset();
        }
    }

    public GameObject Food;
    Rpc rpc;
    Simulation simulation;
    float reward;
    bool finished;
    int step;

    // Start is called before the first frame update
    void Start()
    {
        simulation = GetComponent<Simulation>();
        rpc = new Rpc(this);
    }

    // Update is called once per frame
    void Update()
    {

    }

    public RLResult Step(string action) 
    {
        reward = 0;

        Vector3 direction = Vector3.zero;
        switch(action)
        {
            case "up":
                direction = Vector3.forward;
                break;
            case "down":
                direction = - Vector3.forward;
                break;
            case "right":
                direction = Vector3.right;
                break;
            case "left":
                direction = - Vector3.right;
                break;
        }
        Vector3 newPos = transform.position + direction * 50 * simulation.SimulationStepSize;
        newPos.x = Mathf.Clamp(newPos.x, -50, 50);
        newPos.z = Mathf.Clamp(newPos.z, -50, 50);
        transform.position = newPos;

        simulation.Simulate();
        step += 1;
        if (step >= 1000)
        {
            Debug.Log("Timed out: ending episode");
            finished = true;
        }

        return new RLResult(reward, finished, GetObservation());
    }

    public MyVector3 Reset()
    {
        transform.position = new Vector3(0, 1, 0);
        
        Vector3 newPos = transform.position;
        while ((newPos - transform.position).magnitude < 10.0f)
        {
            newPos = new Vector3(Random.Range(-40.0f, 40.0f), 1.25f, Random.Range(-40.0f, 40.0f));
        }
        Food.gameObject.transform.position = newPos;
        
        finished = false;
        step = 0;

        return GetObservation();
    }

    public MyVector3 GetObservation()
    {
        return new MyVector3(Food.transform.position - transform.position);
    }


    void OnTriggerEnter(Collider other)
    {
        Debug.Log("Yay! The agent got food");
        reward += 1;
        finished = true;
    }
}
