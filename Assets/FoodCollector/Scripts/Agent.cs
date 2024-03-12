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

public class MyQuaternion
{
    public float x;
    public float y;
    public float z;
    public float w;
    public MyQuaternion(Quaternion q)
    {
        this.x = q.x;
        this.y = q.y;
        this.z = q.z;
        this.w = q.w;
    }
    public Quaternion ToQuaternion()
    {
        return new Quaternion(x, y, z, w);
    }
}

public class RLResult
{
    public float reward;
    public bool finished;
    public RayResults observation;
    public RLResult(float reward, bool finished, RayResults observation)
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
            Debug.Log($"you say: {message}");
        }

        [JsonRpcMethod]
        MyVector3 getPosition()
        {
            return new MyVector3(agent.transform.position);
        }

        [JsonRpcMethod]
        MyQuaternion getRotation()
        {
            return new MyQuaternion(agent.transform.rotation);
        }

        [JsonRpcMethod]
        RayResults getObservation()
        {
            return agent.GetObservation();
        }

        [JsonRpcMethod]
        RLResult step(string action) 
        {
            return agent.Step(action);
        }

        [JsonRpcMethod]
        RayResults reset()
        {
            return agent.Reset();
        }
    }

    private Vector3 initialPosition;
    private Quaternion initialRotation;
    public GameObject Food;
    Rpc rpc;
    Simulation simulation;
    float reward;
    bool finished;
    int step;
    RayCasts rayCasts;

    // Start is called before the first frame update
    void Start()
    {
        rpc = new Rpc(this);
        simulation = GetComponent<Simulation>();
        rayCasts = GetComponent<RayCasts>();
        initialPosition = transform.position;
        initialRotation = transform.rotation;
    }

    // Update is called once per frame
    void Update()
    {

    }

    public RLResult Step(string action) 
    {
        reward = 0;

        Vector3 direction = Vector3.zero;
        Quaternion rotation = Quaternion.identity;
        switch(action)
        {
            case "forward":
                direction = transform.forward;
                break;
            case "left":
                rotation = Quaternion.Euler(0, 10, 0);
                break;
            case "right":
                rotation = Quaternion.Euler(0, -10, 0);
                break;
        }

        Vector3 newPos = transform.position + direction * 50 * simulation.SimulationStepSize;
        newPos.x = Mathf.Clamp(newPos.x, -50, 50);
        newPos.z = Mathf.Clamp(newPos.z, -50, 50);
        transform.position = newPos;
        transform.rotation = transform.rotation * rotation;

        simulation.Simulate();
        step += 1;
        if (step >= 1000)
        {
            Debug.Log("Timed out: ending episode");
            finished = true;
        }

        return new RLResult(reward, finished, GetObservation());
    }

    public RayResults Reset()
    {
        transform.position = initialPosition;
        transform.rotation = initialRotation;
        
        Vector3 newPos = transform.position;
        while ((newPos - transform.position).magnitude < 10.0f)
        {
            newPos = new Vector3(Random.Range(-40.0f, 40.0f), 1.25f, Random.Range(-40.0f, 40.0f));
        }
        Food.gameObject.transform.position = newPos;
        
        finished = false;
        step = 0;

        simulation.Simulate();
        
        return GetObservation();
    }

    public RayResults GetObservation()
    {
        return rayCasts.GetObservation();
    }


    void OnTriggerEnter(Collider other)
    {
        Debug.Log("Yay! The agent got food");
        reward += 1;
        finished = true;
    }
}
