import React, { useContext, useState } from 'react'
import axios from 'axios';
import './upload.css'
import { AuthContext } from '../../Hooks/AuthProvider';
import { useNavigate } from 'react-router-dom';


interface Records{
    pname:string;
    dname:string;
    diagnosis:string;
    treatment:string;
}

interface PortProps{
    port:number;
}


const UploadForm:React.FC = () =>{
    const {port} = useContext(AuthContext);
    const url = `http://localhost:${port}`;
    const [data,setData] = useState<Records>({pname:"",dname:"",diagnosis:"",treatment:""});
    const navigate = useNavigate();

    function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
        axios.post(url,data);
        navigate('/');
    }
    function handleInputChange(event: React.ChangeEvent<HTMLInputElement|HTMLTextAreaElement>) {
        const { name, value } = event.target;
        setData({ ...data, [name]: value });
    } 
    return (
        <div>
                <form onSubmit={handleSubmit}>
                    <label>Patient Name</label>
                    <input type = "text" placeholder='Enter patient name here' name = "pname" onChange={handleInputChange}></input>
                    <label>Doctor Name</label>
                    <input type = "text" placeholder='Enter doctor name here' name = "dname" value = {data.dname} onChange={handleInputChange}></input>
                    <label>Diagnosis</label>
                    <input type = 'text' placeholder='Enter Patient Diagnosis here' name = "diagnosis" value = {data.diagnosis} onChange={handleInputChange}></input>
                    <label>Treatment</label>
                    <textarea rows={20} cols={50} placeholder = "Enter Doctor prescibed treatment here" name ="treatment" value = {data.treatment} onChange={handleInputChange}></textarea> 
                    <input type='submit'></input>             
                </form>    
        </div>
    );
}
export default UploadForm;