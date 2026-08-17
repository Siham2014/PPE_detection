import {API_CONFIG} from '../config/api';
export async function checkBackend(){const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),API_CONFIG.timeout);try{await fetch(API_CONFIG.baseUrl+API_CONFIG.healthPath,{signal:controller.signal,mode:'no-cors'});return true}catch{return false}finally{clearTimeout(timer)}}
export async function apiRequest<T>(path:string,options?:RequestInit):Promise<T>{const response=await fetch(API_CONFIG.baseUrl+path,options);if(!response.ok)throw new Error(`API ${response.status}`);return response.json() as Promise<T>}
