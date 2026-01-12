import { useContext, useEffect, useState } from 'react';
import React from 'react';
import './login.css'

import { useNavigate } from 'react-router-dom';
import { AuthContext } from '../../Hooks/AuthProvider';


interface Form{
    server:string;
    uid:string;
    password:string;
};


const LoginForm:React.FC = () =>{
    const [servers,setServers] = useState<Array<string>>([]);
    const [form,setForm] = useState<Form>({server:"",uid:"",password:""});
    const [portmap,setPortmap] = useState<Map<string,number>>(new Map());
    //const [formFlag,setFormFlag] = useState<number>(0);
    const navigate = useNavigate();
    let maP = new Map();
    const {login,setPort} = useContext(AuthContext);
    function handleSubmit(event:React.ChangeEvent<HTMLFormElement>){
        event.preventDefault();
        setPort(String(portmap.get(form.server)));      
        login();
        navigate('/');
    }
    function handleChange(event:React.ChangeEvent<HTMLInputElement|HTMLSelectElement>){
        const {name,value} = event.target;
        setForm({...form,[name]:value});
    }
    useEffect(()=>{
        try
        {
            fetch('http://localhost:3333')
            .then((res) => res.json())
            .then((json)=>{
                setServers(()=>json.map((port:number,index:number) =><option key = {index}>{`Server ${index}`}</option>));
                json.forEach((port:number,index:number) => {
                    maP.set(`Server ${index}`,port);
                });
                setPortmap(maP);
            });
        }
        catch(err){
            console.log(`error = ${err}`)
        }
        },[]);    
    return (
        <div>
            <form onSubmit={handleSubmit}>
                <label htmlFor="servers">Select Server</label>
                <select id='servers' name = "server" onChange={handleChange}>
                    <option value="" selected disabled hidden>Choose</option>
                    {servers}
                </select>
                <label>User ID</label>
                <input type='text' placeholder='Enter your UID' name = "uid" value={form.uid} onChange={handleChange}></input>
                <label>Password</label>
                <input type = 'password' placeholder='Enter your password' name = "password" value = {form.password} onChange={handleChange}></input>
                <input type ="submit"></input>
            </form>
        </div>
    );
}

export default LoginForm;